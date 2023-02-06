# %%
from typing import Iterator

# --- Packages
import pandas as pd
import numpy as np
from random import shuffle
import math

#  TORCH
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

from attention.utils import timed_callout
#from data_preparation.standardize_covariates import standardize


# %%
class MyDistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle: bool = True) -> None:
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dataset = dataset
        self.shuffle = shuffle
        self.total_size = len(dataset)  # dropping excess already done in dataset
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))
        indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # then shuffle indices per rank >> WITHIN EACH RANK!
        if self.shuffle:
            shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Length per part of the DistributedSampler group (so for each rank!)"""
        return self.num_samples






# %%
class ModelDataset_CharModel_OOS(Dataset):
    """
    ---- PyTorch Dataset

    Inputs:

    """

    def __init__(
        self,
        filename: str,
        val_pct: float,
        name: str,
        start_test_month: str,
    ):

        """Initialize Dataset"""
        super().__init__()
        self.id_col = "permno"
        self.target_column = "ret_exc_lead1m"
        self.date_col = "date"
        self.benchmark_col = "ret_12_1"
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

        # ---- Load data
        timed_callout(f"Loading {name} data.")
        data = pd.read_parquet(filename)

        # ----- Many characteristics become available after 1973? Should we start then?
        data = data.loc["1973":]

        # NOTE "ret_exc_lead1m" is forward looking, fitting ends thus basically two months before EOY
        # NOTE TIWI: Should be changed in a future version as testing is now from Nov to Nov, which makes now sense
        # NOTE Below "<" ensures however that there is now overlap between training and testing
        end_fit_month = pd.to_datetime(start_test_month) - pd.DateOffset(months=1)
        data = data.reset_index()


        # --- Train/Val/Test split. Random shuffle
        if name == "val":
            data = data[data["date"] < end_fit_month] # NOTE strictly smaller thus no overlap between training and testing
            unique_dates = pd.Series(data["date"].unique()).sort_values()
            val_dates = unique_dates.iloc[-int(len(unique_dates) * val_pct):]
            data = data[data["date"].isin(val_dates)]

        elif name == "train":
            data = data[data["date"] < end_fit_month]
            unique_dates = pd.Series(data["date"].unique()).sort_values()
            train_dates = unique_dates.iloc[:-int(len(unique_dates) * val_pct)]
            data = data[data["date"].isin(train_dates)]

        else:  # TESTING: 1 Year
            data = data[data["date"] >= end_fit_month]
            end_test_month = end_fit_month + pd.DateOffset(months=12)
            data = data[data["date"] < end_test_month]

        # --- Get characteristic columns
        self.char_cols = [
            c for c in data.columns
            if c not in [self.id_col, self.target_column, self.date_col]
            if "r_-" not in c
        ]

        self.return_cols = [
            c for c in data.columns if "r_-" in c
        ]

        # Chars are in Groups, thus no normalization needed
        chars = data.set_index("date")[self.char_cols].copy()

        # Shall we normalise the historic returns?
        return_chars = data.set_index("date")[self.return_cols].copy()
        
        # Normalise target
        target = data.set_index("date")[self.target_column].copy()
        target_norm = (target - target.groupby("date").mean()) / target.groupby("date").std()

        # --- Pytorch needs numpy
        target = target.to_numpy().astype("float32")
        target_norm = target_norm.to_numpy().astype("float32")
        chars = chars.to_numpy().astype("float32")
        return_chars = return_chars.to_numpy().astype("float32")
        ids = data[self.id_col].to_numpy().astype("int32")
        dates = data[self.date_col].to_numpy().astype("int64")
        benchmark = data[self.benchmark_col].to_numpy().astype("float32")

        # ---- Get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = data.shape[0]
        if self.N % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((self.N - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.N / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.N = self.total_size
        timed_callout(f"Number of samples per rank: {self.num_samples}")

        # --- Crop data
        target = target[-self.N :]
        target_norm = target_norm[-self.N :]
        return_chars = return_chars[-self.N :]
        ids = ids[-self.N :]
        chars = chars[-self.N :]
        dates = dates[-self.N :]
        benchmark = benchmark[-self.N :]

        # Create equal input shape on each rank
        target = target[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        target_norm = target_norm[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        ids = ids[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        chars = chars[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return_chars = return_chars[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        dates = dates[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        benchmark = benchmark[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # --- Create tensors
        self.target = torch.as_tensor(target)
        self.target_norm = torch.as_tensor(target_norm)
        self.chars = torch.as_tensor(chars)
        self.return_chars = torch.as_tensor(return_chars)
        self.ids = torch.as_tensor(ids)
        self.dates = torch.as_tensor(dates)
        self.benchmark = torch.as_tensor(benchmark)

    def __len__(self):
        """Total lenght of current Dataset"""
        return self.N

    def __getitem__(self, idx):
        """Get next item"""
        idx = idx - self.rank * self.num_samples
        return (
            self.chars[idx],
            self.return_chars[idx],
            self.target_norm[idx],
            self.target[idx],
            self.ids[idx],
            self.dates[idx],
            self.benchmark[idx],
        )

    def retrieve_column_names(self):
        return self.char_cols, self.return_cols

# %%
class ModelDataset_CharModel(Dataset):
    """
    ---- PyTorch Dataset

    Inputs:

    """

    def __init__(
        self,
        filename: str,
        val_pct: float,
        name: str,
        seed: int = 123,
    ):

        """Initialize Dataset"""
        super().__init__()
        self.id_col = "permno"
        self.target_column = "ret_exc_lead1m"
        self.date_col = "date"
        self.benchmark_col = "ret_12_1"
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

        # ---- Load data
        timed_callout(f"Loading {name} data.")
        data = pd.read_parquet(filename)

        #data = data.dropna()

        # ----- Many characteristics become available after 1973? Should we start then?
        data = data.loc["1973":]
        data = data.reset_index()
        unique_dates = pd.Series(data["date"].unique())

        # --- Train/Val/Test split. Random shuffle
        if name == "val":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            data = data[data["date"].isin(val_dates)]
        elif name == "train":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            train_dates = unique_dates[~unique_dates.isin(val_dates)]
            data = data[data["date"].isin(train_dates)]
        else:  # FULL Sample
            timed_callout("Using Full Sample.")

        # --- Get characteristic columns
        self.char_cols = [
            c for c in data.columns
            if c not in [self.id_col, self.target_column, self.date_col]
            if "r_-" not in c
        ]

        self.return_cols = [
            c for c in data.columns if "r_-" in c
        ]

        # Chars are in Groups, thus no normalization needed
        chars = data.set_index("date")[self.char_cols].copy()

        # Shall we normalise the historic returns?
        return_chars = data.set_index("date")[self.return_cols].copy()
        
        # Normalise target
        target = data.set_index("date")[self.target_column].copy()
        target_norm = (target - target.groupby("date").mean()) / target.groupby("date").std()

        # --- Pytorch needs numpy
        target = target.to_numpy().astype("float32")
        target_norm = target_norm.to_numpy().astype("float32")
        chars = chars.to_numpy().astype("float32")
        return_chars = return_chars.to_numpy().astype("float32")
        ids = data[self.id_col].to_numpy().astype("int32")
        dates = data[self.date_col].to_numpy().astype("int64")
        benchmark = data[self.benchmark_col].to_numpy().astype("float32")

        # ---- Get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = data.shape[0]
        if self.N % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((self.N - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.N / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.N = self.total_size
        timed_callout(f"Number of samples per rank: {self.num_samples}")

        # --- Crop data
        target = target[-self.N :]
        target_norm = target_norm[-self.N :]
        return_chars = return_chars[-self.N :]
        ids = ids[-self.N :]
        chars = chars[-self.N :]
        dates = dates[-self.N :]
        benchmark = benchmark[-self.N :]

        # Create equal input shape on each rank
        target = target[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        target_norm = target_norm[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        ids = ids[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        chars = chars[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return_chars = return_chars[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        dates = dates[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        benchmark = benchmark[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # --- Create tensors
        self.target = torch.as_tensor(target)
        self.target_norm = torch.as_tensor(target_norm)
        self.chars = torch.as_tensor(chars)
        self.return_chars = torch.as_tensor(return_chars)
        self.ids = torch.as_tensor(ids)
        self.dates = torch.as_tensor(dates)
        self.benchmark = torch.as_tensor(benchmark)

    def __len__(self):
        """Total lenght of current Dataset"""
        return self.N

    def __getitem__(self, idx):
        """Get next item"""
        idx = idx - self.rank * self.num_samples
        return (
            self.chars[idx],
            self.return_chars[idx],
            self.target_norm[idx],
            self.target[idx],
            self.ids[idx],
            self.dates[idx],
            self.benchmark[idx],
        )

    def retrieve_column_names(self):
        return self.char_cols, self.return_cols

# %%
class ModelDataset(Dataset):
    """
    ---- PyTorch Dataset

    Inputs:

    """

    def __init__(
        self,
        filename: str,
        val_pct: float,
        name: str,
        std_char: bool,
        seed: int = 123,
    ):

        """Initialize Dataset"""
        super().__init__()
        self.id_col = "permno"
        self.target_column = "ret_exc_lead1m"
        self.date_col = "date"
        self.benchmark_col = "ret_12_1"
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

        # ---- Load data
        timed_callout(f"Loading {name} data.")
        data = pd.read_parquet(filename)

        # ----- WEEKLY?


        # ------------------------------------
        #data = data.loc["2010":]

        data = data.reset_index()
        unique_dates = pd.Series(data["date"].unique())

        # --- Train/Val/Test split. Random shuffle
        if name == "val":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            data = data[data["date"].isin(val_dates)]
        elif name == "train":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            train_dates = unique_dates[~unique_dates.isin(val_dates)]
            data = data[data["date"].isin(train_dates)]
        else:  # FULL Sample
            timed_callout("Using Full Sample.")

        # --- Get characteristic columns
        self.char_cols = [
            c for c in data.columns if c not in [self.id_col, self.target_column, self.date_col, self.benchmark_col]
        ]


        chars = data.set_index("date")[self.char_cols].copy()
        if std_char:
            chars = (chars - chars.groupby("date").mean() ) / chars.groupby("date").std()

        target = data.set_index("date")[self.target_column].copy()
        target_norm = (target - target.groupby("date").mean()) / target.groupby("date").std()

        # --- Pytorch needs numpy
        target = target.to_numpy().astype("float32")
        target_norm = target_norm.to_numpy().astype("float32")
        chars = chars.to_numpy().astype("float32")

        
        ids = data[self.id_col].to_numpy().astype("int32")
        dates = data[self.date_col].to_numpy().astype("int64")
        benchmark = data[self.benchmark_col].to_numpy().astype("float32")

        # ---- Get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = data.shape[0]
        if self.N % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((self.N - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.N / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.N = self.total_size
        timed_callout(f"Number of samples per rank: {self.num_samples}")

        # --- Crop data
        target = target[-self.N :]
        target_norm = target_norm[-self.N :]
        ids = ids[-self.N :]
        chars = chars[-self.N :]
        dates = dates[-self.N :]
        benchmark = benchmark[-self.N :]

        # Create equal input shape on each rank
        target = target[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        target_norm = target_norm[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        ids = ids[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        chars = chars[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        dates = dates[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        benchmark = benchmark[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]

        # --- Create tensors
        self.target = torch.as_tensor(target)
        self.target_norm = torch.as_tensor(target_norm)
        self.chars = torch.as_tensor(chars)
        self.ids = torch.as_tensor(ids)
        self.dates = torch.as_tensor(dates)
        self.benchmark = torch.as_tensor(benchmark)

    def __len__(self):
        """Total lenght of current Dataset"""
        return self.N

    def __getitem__(self, idx):
        """Get next item"""
        idx = idx - self.rank * self.num_samples
        return (
            self.chars[idx],
            self.target_norm[idx],
            self.target[idx],
            self.ids[idx],
            self.dates[idx],
            self.benchmark[idx],
        )

    def retrieve_column_names(self):
        return self.char_cols


# %%

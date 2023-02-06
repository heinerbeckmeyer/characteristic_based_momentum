# %%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from attention.model import SingleHeadAttention
from attention.utils import timed_callout


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
        seed: int = 123,
    ):

        """ Initialize Dataset """
        super().__init__()
        self.id_col = "permno"
        self.target_column = "ret_exc_lead1m"
        self.date_col = "date"
        self.benchmark_col = "ret_12_1"

        

        # ---- Load data
        timed_callout(f"Loading {name} data.")
        data = pd.read_parquet(filename)
        
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
        else: # FULL Sample
            timed_callout("Using Full Sample.")

        # --- Get characteristic columns
        self.char_cols = [c for c in data.columns if c not in [self.id_col, self.target_column, self.date_col, self.benchmark_col]]

        # --- Pytorch needs numpy
        target = data[self.target_column].to_numpy().astype("float32")
        # NOTE Scale daily returns to monthly returns to be on same level as target
        chars = (data[self.char_cols] * 21.0).to_numpy().astype("float32")
        ids = data[self.id_col].to_numpy().astype("int32")
        dates = data[self.date_col].to_numpy().astype("int64")
        benchmark = data[self.benchmark_col].to_numpy().astype("float32")



        # ---- Get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = data.shape[0]

        # --- Create tensors
        self.target = torch.as_tensor(target)
        self.chars = torch.as_tensor(chars)
        self.ids = torch.as_tensor(ids)
        self.dates = torch.as_tensor(dates)
        self.benchmark = torch.as_tensor(benchmark)
    

    def __len__(self):
        """ Total lenght of current Dataset"""
        return self.N

    def __getitem__(self, idx):
        """ Get next item """
        
        return (
            self.chars[idx],
            self.target[idx],
            self.ids[idx],
            self.dates[idx],
            self.benchmark[idx],
        )
# %%
def train_epoch(
    dataLoader,
    model,
    loss_fn,
    optimizer,
    epoch: int = 0,
):
    device = torch.device("cpu")
    nobs = len(dataLoader)

    print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 1])).astype("int")
    #print_batches = [nobs-1]


    # --- Set model to train mode
    model.train()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    for batch,(chars, target, _, _, _) in enumerate(dataLoader):

        # ---1: Send input data to device
        chars = chars.to(device)

        # ---2: Run the model
        prediction, _= model.forward(chars)

        # ---3: Calculate the loss
        loss = loss_fn(prediction,target)

        # ---4: Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # ---5: Update parameters   
        optimizer.step()

        # ---- Record statistics
        running_loss += loss.item() * target.numel()
        running_N += target.numel()
        running_mse += ((prediction-target)).pow(2).sum().item()
        running_benchmark += ((target.mean()-target)).pow(2).sum().item()

        if batch in print_batches:
            timed_callout(
                f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                + f"Total={running_loss/running_N:.5f} --- "
                + f"R2={1-running_mse/ running_benchmark:.5f}"
            )
        
    return running_loss/running_N, 1-running_mse/running_benchmark

# %%
def val_epoch(
    dataLoader,
    model,
    loss_fn,
    epoch: int = 0,
):
    device = torch.device("cpu")
    nobs = len(dataLoader)

    print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 1])).astype("int")
    #print_batches = [nobs-1]


    # --- Set model to eval mode
    model.eval()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    with torch.no_grad():
        for batch,(chars, target, _, _,_) in enumerate(dataLoader):
            # ---1: Send input data to device
            chars = chars.to(device)

            # ---2: Run the model
            prediction, _ = model.forward(chars)

            # ---3: Calculate the loss
            loss = loss_fn(prediction,target)

            # ---- Record statistics
            running_loss += loss.sum().item() * target.numel()
            running_N += target.numel()
            running_mse += ((prediction-target)).pow(2).sum().item()
            running_benchmark += ((target.mean()-target)).pow(2).sum().item()

            if batch in print_batches:
                timed_callout(
                    f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                    + f"Total={running_loss/running_N:.5f} --- "
                    + f"R2={1-running_mse/ running_benchmark:.5f} "
                )
        
    return running_loss/running_N, 1-running_mse/running_benchmark


# %%
config = {}
config["model_params"] = dict(
    d_embedding = 8,
    n_lags = 231,
    alpha=1.0,
    retrieve_attention_mask=False,
    do_pos_encoding = True,
    dropout_p = 0.1,
)


config["optim_params"] = dict(lr=0.001, amsgrad=True, weight_decay=0.1)

max_epochs = 50 

train_data = ModelDataset(
    filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
    val_pct=0.3,
    name="train"
)
val_data = ModelDataset(
    filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
    val_pct=0.3,
    name="val"
)

#batch_size = 8**5
batch_size = 8*round(len(train_data)*0.001) + 8

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

# %%

loss_fn = nn.MSELoss()

model = SingleHeadAttention(**config["model_params"])
optimizer = torch.optim.AdamW(model.parameters(),**config["optim_params"])

# %%
train_loss = []
val_loss = []
train_r2 = []
val_r2 = []


for epoch in range(max_epochs):

   # ---- Training 
    tmp_loss, tmp_r2 = train_epoch(
        dataLoader=train_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epoch=epoch)
    train_loss.append(tmp_loss)
    train_r2.append(tmp_r2)

    # ---- Validation
    tmp_loss, tmp_r2= val_epoch(
        dataLoader=val_loader,
        model=model,
        loss_fn=loss_fn,
        epoch=epoch)
    val_loss.append(tmp_loss)
    val_r2.append(tmp_r2)


    # ---- Plot
    if epoch%5==0:
        fig,ax=plt.subplots(2,1,sharex=True)
        ax[0].plot(np.arange(len(train_loss)),train_loss,"-ob",label="Training")
        ax[0].plot(np.arange(len(val_loss)),val_loss,"-or",label="Validation")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("")

        ax[1].plot(np.arange(len(train_r2)),train_r2,"-ob",label="Training")
        ax[1].plot(np.arange(len(val_r2)),val_r2,"-or",label="Validation")
        ax[1].set_ylabel(r"$R^2$")
        ax[1].set_xlabel("Epoch")
        plt.pause(0.5)


# %%

# %%
import numpy as np


# Standardize covariates
def standardize(
    characteristics, standardization_type: str = "standardize", spare_cols: list = [], n_groups: int = None
):
    """standardizes characteristics.

    Args:
        characteristics (pd.DataFrame): characteristics table
        permnos (pd.Series): permnos in same order as characteristics.
        standardization_type (str, optional): Defaults to "standardize".
            May be either "standardize" for normalization, "minmax", or "rank" for mapping
            to a range of [-1, 1] cross-sectionally.
            NEW: "hist" for portfolio assignments
        spare_cols (list, optional): Columns not to be standardized.
    """

    to_spare = characteristics[spare_cols]
    rank = characteristics.groupby("date").rank(method="dense")
    M = rank.groupby("date").transform("max")  # highest rank, i.e. max number of variation

    # standardize
    if standardization_type == "standardize":
        sd = characteristics.groupby("date").std()
        mn = characteristics.groupby("date").mean()
        characteristics = (characteristics - mn) / sd

    elif standardization_type == "rank":
        characteristics = 2 * (rank.sub(1).divide(M.sub(1), axis=0)) - 1

    elif standardization_type == "hist":
        if n_groups is None:
            raise ValueError("With standardization_type='hist', you have to specify n_groups as an integer.")
        characteristics = np.ceil((rank.sub(1).divide(M.sub(1), axis=0) + 1e-9) * n_groups)
        characteristics[characteristics > n_groups] = n_groups

    elif standardization_type == "minmax":
        emp_min = characteristics.groupby("date").min()
        emp_max = characteristics.groupby("date").max()
        characteristics = (characteristics - emp_min) / (emp_max - emp_min) * (1 - (-1)) + (-1)

    else:
        raise ValueError("Wrong standardization type.")

    characteristics[spare_cols] = to_spare

    return characteristics


# %%

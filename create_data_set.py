# %%
"""
    Creates data set using:
        - the dataset of Gu, Kelly, Xiu (2020, RFS).
        - WRDS/CRSP 
    
    For WRDS:
        Must be run in terminal when used for the first time to avoid password query

        1. db = wrds.Connection()
            -->Enter username and password
        2. db.create_pgpass_file()

"""




# %%
import wrds
import pandas as pd
import numpy as np

from data_preparation.standardize import standardize


# --- Declaration of variables
crsp_database_type = "d"
return_column = "ret_exc_lead1m"
merge_target_only = False
standardization_type = "ranks"



# %%
# --------------------------------------- Kelly  ----------------------------------------------------------
print("Loading JKP Data.")
# NOTE: We actually do not need the chars yet. Just the future returns. Keep ip for future research though.
if merge_target_only:
    characteristics = pd.read_parquet(
        "../03_data/kelly_characteristics/jkp.pq", columns=[ return_column, "ret_12_1"])
else:
    characteristics = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq")
    
    # ---- Delete seasonal returns
    seas_cols = [c for c in characteristics.columns if "seas_" in c]
    characteristics = characteristics[[c for c in characteristics.columns if "seas_" not in c]]

    characteristics = characteristics.reset_index()
    characteristics = characteristics.groupby("date").apply(
        lambda x: standardize(x,
            standardization_type=standardization_type,
            n_groups=100,
            spare_cols=[return_column, "permno", "date"]
            )
    ) 

    characteristics.loc[:, characteristics.columns != 'ret_12_1'] = characteristics.loc[:, characteristics.columns != 'ret_12_1'].fillna(0)

    characteristics = characteristics.set_index(["date", "permno"])

# NOTE: I keep Kelly's "momentum" as benchmark later on
characteristics = characteristics.dropna(subset=["ret_12_1", return_column])



# %%
# --------------------------------------- CRSP ----------------------------------------------------------
db = wrds.Connection(wrds_username="twied03")
print("Loading CRSP Data.")
# NOTE Mayby we should adjust for delistings at some point? 
# Predicting returns provided by kelly though.
# ---  Prices and returns
start_date = characteristics.index.get_level_values("date").unique().min().date()
crsp_prices = db.raw_sql(
    f"""
        select distinct     a.permno,
                            a.date,
                            a.ret
        from
                crsp.{crsp_database_type}sf as a
        where
                        a.date >= '{start_date}'
    """
)

# ---- specify format
crsp_prices.date = pd.to_datetime(crsp_prices.date, format="%Y-%m-%d")
dtype = {
    "permno": "i4",
}
crsp_prices = crsp_prices.astype(dtype)

# --- Close db
db.close()


# --------------------------------------- Merge & Create data set ----------------------------------------------------------
print("Creating Data Set.")
crsp_prices = crsp_prices.sort_values("date")


# Get EOM dates
all_eom_dates = (
    crsp_prices.groupby([crsp_prices["date"].dt.year, crsp_prices["date"].dt.month])
    .last()["date"].reset_index(drop=True).to_frame("eom_date")
    )
all_eom_dates["merge_date"] = all_eom_dates.set_index(["eom_date"]).shift(1, freq="MS").index.values

# We need a history of at least 252 days
all_eom_dates = all_eom_dates[
    all_eom_dates.eom_date >= crsp_prices["date"].unique()[252]
]

# Unstack return data
crsp_prices = crsp_prices.set_index(["date", "permno"]).sort_index().unstack()


final_data = pd.DataFrame()
for eom_date, merge_date in zip(all_eom_dates.eom_date, all_eom_dates.merge_date):
    print(f"\rWorking on Date: {merge_date}", end="", flush=True)
    
    # NOTE Shift by +1 as end_index is exclusive
    end_idx = np.where(crsp_prices.index == eom_date)[0][0] + 1
    start_idx = end_idx - (252 - 21)  
    current_data = crsp_prices.iloc[start_idx:end_idx]

    current_data = current_data.ret.dropna(axis=1, how="all")
    current_data = current_data.fillna(0)
    current_data.index = [f"r_{i:.0f}" for i in np.arange(-252, -21)]
    current_data = current_data.T

    current_data["date"] = merge_date
    current_data = current_data.reset_index().set_index(["date", "permno"])
    
    # Calculate log returns
    current_data = np.log(1+current_data)

    # Merge
    tmp_characteristics = characteristics[
        characteristics.index.get_level_values("date") == merge_date
        ]
    current_data = current_data.merge(tmp_characteristics, left_index=True, right_index=True, how="right")

    # Concat 
    final_data = pd.concat([final_data, current_data])

# Saving
print("\nSaving")
final_data.to_parquet(f"../04_results/Momentum_DataSet_TargetOnly={merge_target_only}_std={standardization_type}.pq")
print("Done")








# %%

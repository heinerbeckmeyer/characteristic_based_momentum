# %%
import pandas as pd
import numpy as np
import wrds
import os

# %% Additional Functions
def vw(x, weights):
    out = np.sum(x * weights) / np.sum(weights)
    return out



# %%
# ----------------------- Get CRSP daily returns ---------------------
# NOTE Should already have been downloaded in "analysis_oos.py"
fn = "../03_data/crsp_trading_dates.pq"
if os.path.exists(fn):
    crsp = pd.read_parquet(fn)
else:
    db = wrds.Connection(wrds_username="twied03")
    print("Loading CRSP Data.")
    crsp = db.raw_sql(
        f"""
            select distinct     a.permno,
                                a.date,
                                a.ret
            from
                    crsp.dsf as a
            where
                            a.date >= '01/01/1970'
        """
    )

    # ---- specify format and save
    crsp["date"] = pd.to_datetime(crsp["date"], format="%Y-%m-%d")
    crsp = crsp.astype({"permno": "i4"})
    crsp = crsp.sort_values("date").reset_index(drop=True)
    crsp.to_parquet(fn)

    # --- Close db
    db.close()

# %%
# ----------------------- Get ret_12_1 from Kelly ---------------------
jkp = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq", columns=["ret_12_1", "market_equity"])
jkp = jkp["1970":]

# Shift MCAP/RET_12_1 
jkp.loc[:,"mcap_lag"] = jkp.groupby("permno")["market_equity"].shift()
jkp.loc[:,"ret_12_1_lag"] = jkp.groupby("permno")["ret_12_1"].shift()

jkp = jkp.dropna()

# Calc decie portfolios
jkp.loc[:,"pf"] = jkp.groupby("date")["ret_12_1"].transform(lambda x: pd.qcut(x, q=10, labels=False))

# Merge
crsp.loc[:,"merge_date"] = crsp["date"] - pd.TimedeltaIndex(crsp["date"].dt.day.values - 1, unit="d")

data = crsp.merge(jkp, left_on=["merge_date", "permno"], right_index=True)
data = data.dropna(subset=["pf", "ret"])
# %%
# ----------------------- Calc daily WML ---------------------
pf_ret = data.groupby(["date", "pf"]).apply(lambda x: vw(x["ret"], x["mcap_lag"])).to_frame("ret")

wml = (
    pf_ret.reset_index().groupby("date")
    .apply(lambda x: (x[x["pf"] == 9].ret.values - x[x["pf"] == 0].ret.values)[0])
    .to_frame(name="ret")
)

# %%
# ----------------------- Calc monthly variance forecast ---------------------
# NOTE We need to shift datetimes to be on par with JKP data as we use "ret_exc_lead1m".
# Said differently WML in t uses ret_12_1 from t and ret_exc_lead1m which is in t+1
# Here we use ret_12_1 from t-1 and ret which is from t

wml.index = wml.index - pd.DateOffset(months=1) # TIWI: Checked, should be correct

# Get End-of-Month dates
eom_dates = wml.groupby(pd.Grouper(freq="M")).last().index.tolist()

sigma = pd.DataFrame()
for dt in eom_dates:
    tmp = wml[:dt]

    # Need at least 126 days lookback
    if tmp.shape[0]<126:
        continue
    else:
        tmp = tmp[-126:]

        out = pd.DataFrame({
            "sigma2": 21/126 *np.sum(tmp.ret.pow(2)),
            "date": dt
        }, index=[0])
        sigma = pd.concat([sigma, out])

sigma = sigma.reset_index(drop=True)

# Shift to next month start to make it a forecast
sigma["date"] = sigma["date"] - pd.TimedeltaIndex(sigma["date"].dt.day.values - 1, unit="d")
sigma["date"] = sigma["date"] + pd.DateOffset(months=1) 

sigma.to_parquet("../03_data/wml_sigma2_forecast.pq")
# %%

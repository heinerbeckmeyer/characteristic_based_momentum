#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


#%%
# ---- Attention weights
model_name = "CharWeighted_posEnc=True___20230105_2023"

att = pd.read_parquet(f"../05_models/{model_name}/results/attention_weights.pq")

att = att.set_index(["date", "id"])
att = att.groupby("date").mean()

# True values

# Average weight
fig, ax = plt.subplots()
x = np.arange(-att.shape[1] - 21,-21)

# Sinus
true_val = np.sin(np.arange(len(x))*2*np.pi/len(x)) + 2
true_val /= true_val.sum()

 # Exponential
# true_val = np.exp(0.02*np.arange(len(x)))
# true_val /= np.sum(true_val)

ax.plot(x, att.mean(), ls="-", marker=".", color="b")
#ax.plot(x, true_val, color = "k")
ax.axhline(att.mean().mean(), ls="--", color="k")

ax.set_xlabel("Lag (Days)")
ax.set_ylabel("Attention weight")


# %%
# ---- Additional function
def vw(x, weights):
    out = np.sum(x * weights) / np.sum(weights)
    return out


# ------------------------------
# Get Data
data = pd.read_parquet(f"../05_models/{model_name}/results/predicted_values.pq")

# Get Market Cap for weights
mcap = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq", columns=["market_equity"])
mcap = mcap.sort_index().groupby("permno")["market_equity"].shift().to_frame()
mcap = mcap.dropna()

# Form momentum portfolios based on prediction
data["pf"] = data.groupby("date")["prediction"].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop"))

# Form momentum portfolios based on benchmark
data["pf_benchmark"] = data.groupby("date")["benchmark"].transform(lambda x: pd.qcut(x, q=10, labels=False))

# Merge
data = data.rename(columns={"id": "permno"})
data = data.set_index(["date", "permno"])
data = data.merge(mcap, left_index=True, right_index=True, how="left")

# Calculate returns
ret_benchmark = (
    data.groupby(["date", "pf_benchmark"])
    .apply(lambda x: vw(x["target"], x["market_equity"]))
    .to_frame("ret_benchmark")
)
ret_model = data.groupby(["date", "pf"]).apply(lambda x: vw(x["target"], x["market_equity"])).to_frame("ret_model")

# Momentum portfolio
wml_benchmark = (
    ret_benchmark.reset_index()
    .groupby("date")
    .apply(
        lambda x: (x[x["pf_benchmark"] == 9].ret_benchmark.values - x[x["pf_benchmark"] == 0].ret_benchmark.values)[0]
    )
    .to_frame(name="ret_wml_benchmark")
)

wml = (
    ret_model.reset_index()
    .groupby("date")
    .apply(lambda x: (x[x["pf"] == 9].ret_model.values - x[x["pf"] == 0].ret_model.values)[0])
    .to_frame(name="ret_wml")
)

# %%
# ----- Cumulative Return
start = "1980-01-01"
end = "1987-01-01"
fig, ax = plt.subplots()
ax.plot(wml_benchmark.loc[start:end].index, (wml_benchmark.loc[start:end] + 1).cumprod().ret_wml_benchmark, label="Benchmark", color="k")

ax.plot(wml.loc[start:end].index, (wml.loc[start:end] + 1).cumprod().ret_wml, label="Model", color="b")

ax.set_xlabel("Date")
ax.set_ylabel("Dollar value of investment")

gain = (wml.loc[start:end] + 1).prod().ret_wml / (wml_benchmark.loc[start:end] + 1).prod().ret_wml_benchmark
sr_model = (np.sqrt(12) * wml.loc[start:end].mean()/wml.loc[start:end].std()).values[0]
sr_bench = (np.sqrt(12) * wml_benchmark.loc[start:end].mean()/wml_benchmark.loc[start:end].std()).values[0]
std_gain = wml.loc[start:end].std().values[0]/wml_benchmark.loc[start:end].std().values[0]
ax.set_title(f"GAIN: {gain:.2f}   -   SR: {sr_model:.3f}   -   B-SR: {sr_bench:.3f}  -  STD-Gain: {std_gain:.2f} ")
plt.yscale("log")
plt.legend()








#%%
# Intermediate momentum?
intermediate_cols = [f"r_{i}" for i in range(-252, -125)]
other_cols = [c for c in att.columns if c not in intermediate_cols]

intermediate_data = (att[intermediate_cols].sum(axis=1) / len(intermediate_cols)) /(att[other_cols].sum(axis=1) / len(
    other_cols)
)

# intermediate_data /= att[other_cols].sum(axis=1) / len(other_cols)
# intermediate_data *= 100

fig, ax = plt.subplots()
ax.plot(intermediate_data.groupby(intermediate_data.index.get_level_values(0)).mean())
ax.axhline(1, ls="--", color="k")
# %%

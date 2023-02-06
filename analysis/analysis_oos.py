# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
import re
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib
import wrds
import statsmodels.api as sm
import itertools
from pandas_datareader.famafrench import get_available_datasets

from pylatex import Math, Tabularx, MultiColumn, LongTable, Tabular
from pylatex.utils import NoEscape

# %%
#model_name = "_OOS_models_CharWeighted_20230106_1005"
#model_name = "_OOS_models_CharWeighted_20230113_1408"
model_name = "_OOS_models_CharWeighted_20230120_1604"
results_loc = "../05_models"
barroso_target = 12 / 100 / np.sqrt(12)     # NOTE Target volatility of 12% p.a.

NSIZE = 10
NSIZE_EXCLUDE = 2   # Excludes Portfolios <=NSIZE_EXCLUDE

CORES = os.cpu_count()//2
ENSEMBLE = False
N_ENSEMBLE = 5


INT_ENSEMBLE = N_ENSEMBLE if ENSEMBLE else 1

saveLoc = os.path.join(results_loc, model_name, "__results")
os.makedirs(saveLoc, exist_ok="True")

# ---- Additional function
def vw(x, weights):
    out = np.sum(x * weights) / np.sum(weights)
    return out

def math(x):
    return Math(data=[NoEscape(x)], inline=True)


def read_ff(dataset_name, verbose=False):
    """Reads Fama/French datasets."""

    # get the number of rows to skip.
    read = False
    skipped_rows = 0
    while read is False:
        try:
            data = pd.read_csv(
                ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
                skiprows=skipped_rows,
                index_col=0,
                nrows=1000,
            )
            read = True
        except (pd.errors.ParserError, IndexError, UnicodeDecodeError) as e:
            if verbose:
                print(e, ", skipping an additional row.")
            skipped_rows += 1

    # get row of first "real" date:
    if (data.index.dtype) == "object":
        dates = data.index.str.extract(r"^(\d{8}|\d{6})", expand=False)
        ii_first_date = np.where(~dates.isnull())[0][0]
        if ii_first_date > 0:
            ii_first_date += 1
    else:
        ii_first_date = 0

    # read in data:
    data = pd.read_csv(
        ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + dataset_name + "_CSV.zip"),
        skiprows=ii_first_date + skipped_rows,
        index_col=0,
    )

    # skip annual data at the bottom:
    annual_data_cutoff = np.where(data.isnull().all(axis=1))[0]
    if len(annual_data_cutoff > 0):
        data = data.iloc[: annual_data_cutoff[0]]

    # check type of data: monthly vs. daily:
    if "daily" in dataset_name:
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
    else:
        data.index = pd.to_datetime(data.index, format="%Y%m").to_period("M")

    # data is given in percentage points:
    data = data.astype("f8")
    data /= 100

    return data


def get_datasets():
    return get_available_datasets()

def drop_head(df, format="%d.%m.%Y"):
    for idx, row in df.iterrows():
        try:
            i = pd.to_datetime(row.iloc[0], format=format) 
        except ValueError:
            i = np.nan
        if not pd.isnull(i):
            break

    df = df[idx:]
    df = df.rename(columns={df.columns[0]: "date"})

    if df.dtypes["date"]=="object":
        df["date"] = pd.to_datetime(df["date"], format=format)

    # Drop empty columns
    df = df.dropna(how="all", axis=1)
    return df.reset_index(drop=True)

def reader(fn, ENSEMBLE, N, attention: bool = False):
    if attention:
        if ENSEMBLE:
            fn = os.path.join(fn, "results", f"Ensemble_N={N}_attention_weights.pq")
        else: 
            fn = os.path.join(fn, "results", "attention_weights.pq")
    else:
        if ENSEMBLE:
            fn  = os.path.join(fn, "results", f"Ensemble_N={N}_predicted_values.pq")
        else:
            fn  = os.path.join(fn, "results", "predicted_values.pq")
    if os.path.exists(fn):
        return pd.read_parquet(fn)
    else:
        return pd.DataFrame()

def colors(n):
    return sns.cubehelix_palette(n, rot=-0.25, light=0.7)


def rgb_color(input: tuple):
    return [i / 255 for i in input]

def crra(r,gamma):
    u = (1 + r)**(1-gamma) / (1-gamma)
    return u

def ce_crra(u, gamma):
    ce = (u * (1 - gamma)) ** (1/(1-gamma)) - 1
    return ce

def calc_MDD(networth):
    df = pd.Series(networth, name="nw").to_frame()
    max_peaks_idx = df.nw.expanding(min_periods=1).apply(lambda x: x.argmax()).fillna(0).astype(int)
    df["max_peaks_idx"] = pd.Series(max_peaks_idx).to_frame()
    nw_peaks = pd.Series(df.nw.iloc[max_peaks_idx.values].values, index=df.nw.index)
    df["dd"] = (df.nw - nw_peaks) / nw_peaks
    df["mdd"] = (
        df.groupby("max_peaks_idx").dd.apply(lambda x: x.expanding(min_periods=1).apply(lambda y: y.min())).fillna(0)
    )
    return df

def norm_func(df, norm_col:str = "market_equity", equal_weights: bool = False):
    if equal_weights:    
        df["w"] = 1/df.shape[0]
    else:
        df["w"] = df[norm_col]/np.sum(df[norm_col].abs())
        df = df.drop(columns=[norm_col])

    return df

def standardize_signal(group):
    ranks = group.rank(method="min")
    ranks = ranks.sub(1).divide(ranks.max() - ranks.min()) - 0.5

    return ranks

def agg_weights(
    w,
    trading_grid,
    window=2,
    ea=pd.DataFrame(), gdp=pd.DataFrame(), cpi=pd.DataFrame(), fomc=pd.DataFrame(),
    r = pd.DataFrame(), mkt = pd.DataFrame()
    ): 
    # --- Additional function
    def sum_weights(data, name, window=window):
        idx = data[data[f"{name}_indicator"]==1].index.tolist()
        

        # Add window if necessary
        if name not in ["pos", "baker", "q5", "qsmall", "qlarge"]:
            all_idx = []
            for i in idx:
                all_idx += list(np.arange(i-window, i+1+window))
        else:
            all_idx = idx
        # --- Check that all idx are within bounds
        all_idx = [idx for idx in all_idx if (idx>=0) & (idx<=data.index.max()) ]

        # --- Check that there is no overlap 
        all_idx = list(set(all_idx))

        # --- Prepare Output
        out = {
            f"w_{name}": data.iloc[all_idx].w.sum() if all_idx else np.nan,
            f"w_non_{name}": data[~data.index.isin(all_idx)].w.sum(),
            f"n_{name}": data.iloc[all_idx][f"{name}_indicator"].sum() if all_idx else 0,
            f"n_w_{name}": len(all_idx),
            f"n_w_non_{name}": np.sum(~data.index.isin(all_idx)),

        }

        return out

    # ------------ Additional cleaning ---------
    # NOTE There seem to be duplicate dates causing trouble in the ea dates data. Drop them.
    ea = ea.drop_duplicates(subset=["anndats_act"])

    # --------------------------
    
    # Get weights per ID and Date
    kelly_date = w.name[0]
    permno = w.name[1]
    lookback = w.shape[0]

    # Get dates to weights
    data = trading_grid[trading_grid<kelly_date].iloc[-lookback:]
    data = data.to_frame("date")
    data["w"] = w.values
    data["permno"] = permno
    data = data.reset_index(drop=True)
    data = data.fillna(0)
    
    # --- Prepare Output
    out = {
        "date": kelly_date,
        "permno": permno,
        # --- Intermediate momentum
        "w_12_7": data.iloc[:126].w.sum(),
        "w_6_1": data.iloc[126:].w.sum(),

    }

    # --- Earnings Announcement
    if not ea.empty:
        ea["ea_indicator"] = 1
        data = data.merge(
            ea.set_index(["anndats_act", "permno"])["ea_indicator"],
            left_on=["date", "permno"],
            right_index=True,
            how="left")
        data = data.fillna(0)
        
        out.update(sum_weights(data, name="ea"))

    # --- GDP
    if not gdp.empty:
        gdp["gdp_indicator"] = 1
        data = data.merge(
            gdp.set_index("date")["gdp_indicator"],
            on="date",      
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="gdp"))
    
    # --- CPI
    if not cpi.empty:
        cpi["cpi_indicator"] = 1
        data = data.merge(
            cpi.set_index("date")["cpi_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="cpi"))
    
    # --- FOMC
    if not fomc.empty:
        fomc["fomc_indicator"] = 1
        data = data.merge(
            fomc.set_index("date")["fomc_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="fomc"))
    
    
    if not r.empty:
        # --- POS/NEG returns
        r = r[r["date"].isin(data["date"])].set_index("date")
        r["pos_indicator"] = (r["ret"]>0).astype(int)
        data = data.merge(
            r["pos_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="pos"))
    
        # --- Large returns
        q5 = r["ret"].abs().quantile(0.95)
        r["q5_indicator"] = (r["ret"].abs()>q5).astype(int)
        data = data.merge(
            r["q5_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="q5"))

        # --- smallest returns
        qsmall = r["ret"].quantile(0.025)
        r["qsmall_indicator"] = (r["ret"]<=qsmall).astype(int)
        data = data.merge(
            r["qsmall_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="qsmall"))

        # ---- largest returns
        qlarge = r["ret"].quantile(0.975)
        r["qlarge_indicator"] = (r["ret"]>=qlarge).astype(int)
        data = data.merge(
            r["qlarge_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="qlarge"))
    
        
    # --- BAKER mkt.abs()>2.5%
    if not mkt.empty:
        mkt["baker_indicator"] = (mkt["mkt"].abs()>0.025).astype(int)
        data = data.merge(
            mkt["baker_indicator"],
            on="date",
            how="left")
        data = data.fillna(0)

        out.update(sum_weights(data, name="baker"))

    
    return pd.DataFrame(out, index=[0])


def align(num_str, neg: bool = True, stars: bool = True):
    
    # Check stars
    if stars:
        nstars = len(re.findall("[*]",num_str))
        num_str = (
            num_str.split("}")[0] + r"\phantom{" + "".join(["*"] * (3-nstars))
            + r"}}"
        )

    # Check if negativ
    if neg:
        if "-" not in num_str:
            num_str = r"\phantom{-}" + num_str
    
    return num_str
###############################################################################
#
#                   Definitions for Plots
#
###############################################################################
landscape_width = 20
landscape_height = 10

cm = 1 / 2.54
width = 15.92 * cm
height = 10 * cm

matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.xmargin"] = 0.02
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.edgecolor"] = "0.15"
matplotlib.rcParams["axes.linewidth"] = 1.25
matplotlib.rcParams["figure.dpi"] = 800
matplotlib.rcParams["savefig.dpi"] = 800
matplotlib.rcParams["savefig.transparent"] = True
matplotlib.rcParams["savefig.format"] = "pdf"
matplotlib.rcParams["mathtext.fontset"] = "stixsans"

predefined_color = [rgb_color((45, 55, 75)), rgb_color((62, 195, 213)), rgb_color((200, 199, 205))]

# %%
###########################################################################
#
#               Data Loading
#
###########################################################################
# Get FF as Benchmarks
ff = read_ff("F-F_Research_Data_5_Factors_2x3")
ff.index.name = "date"
ff = ff.sort_index()
ff = ff.drop(columns=["RF"])
mom = read_ff("F-F_Momentum_Factor")
mom.index.name = "date"
ff = ff.merge(mom, on="date")
ff.index = ff.index.to_timestamp()
ff.columns = [c.replace(" ", "") for c in ff.columns]

# Get daily market
mkt_daily = read_ff("F-F_Research_Data_Factors_daily")
mkt_daily.index.name = "date"
mkt_daily = mkt_daily["Mkt-RF"].to_frame(name="mkt")
mkt_daily = mkt_daily["1960":]

# --- MACRO/EA Dates
# EA
ea = pd.read_parquet("../03_data/ibes_announcement_dates.pq")

# fomc
fomc = pd.read_excel("../03_data/announcements_full_sample_v2.xlsx", sheet_name="FOMC")
fomc = drop_head(fomc)
# GDP
gdp = pd.read_excel("../03_data/announcements_full_sample_v2.xlsx", sheet_name="GDP")
gdp = drop_head(gdp, format="%Y-%m-%d")

# CPI
cpi = pd.read_excel("../03_data/announcements_full_sample_v2.xlsx", sheet_name="CPI")
cpi = drop_head(cpi, format="%Y-%m-%d")


# Get Model Data
folder_list = glob.glob(
    os.path.join(results_loc, model_name, "*")
)

# Concat results
data = Parallel(os.cpu_count()//2, backend="loky", verbose=True)(
    delayed(reader)(fn, ENSEMBLE=ENSEMBLE, N=N_ENSEMBLE) for fn in folder_list
)
data = pd.concat(data)
data = data.reset_index(drop=True).sort_values("date")

# Form momentum portfolios based on prediction
data["pf"] = data.groupby("date")["prediction"].transform(lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop"))

# Form momentum portfolios based on benchmark
data["pf_benchmark"] = data.groupby("date")["benchmark"].transform(lambda x: pd.qcut(x, q=10, labels=False))


# Get Market Cap for weights and merge
mcap = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq", columns=["market_equity"])
mcap = mcap.sort_index().groupby("permno")["market_equity"].shift().to_frame()
mcap = mcap.dropna()

data = data.rename(columns={"id": "permno"})
data = data.set_index(["date", "permno"])
data = data.merge(mcap, left_index=True, right_index=True, how="left")

# Form size Protfolios
data["pf_size"] = data.groupby("date")["market_equity"].transform(lambda x: pd.qcut(x, q=NSIZE, labels=False))
data["size_percentile"] = data.groupby("date")["market_equity"].transform(lambda x: pd.qcut(x, q=100, labels=False))

# Portfolio (Long-Short) by Prediction
data["pf_prediction"] = 0
data.loc[data.prediction >=0, "pf_prediction"] = 1

# Standardize signal
data.loc[:,"signal"] = data.groupby("date")["prediction"].transform(lambda x: standardize_signal(x))


# --------------------- Other Benchmarks ---------------------------------
# Get Benchmarks from Kelly
kelly_col = ["ret_6_1", "ret_12_7", "prc_highprc_252d"]
bm_data = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq", columns=kelly_col)
data = data.merge(bm_data, on=["date", "permno"], how="left")


# Calculate returns
ret_benchmark = (
    data.groupby(["date", "pf_benchmark"])
    .apply(lambda x: vw(x["target"], x["market_equity"]))
    .to_frame("ret_benchmark")
)
ret_model = data.groupby(["date", "pf"]).apply(lambda x: vw(x["target"], x["market_equity"])).to_frame("ret_model")

# SM
wml_benchmark = (
    ret_benchmark.reset_index()
    .groupby("date")
    .apply(
        lambda x: (x[x["pf_benchmark"] == 9].ret_benchmark.values - x[x["pf_benchmark"] == 0].ret_benchmark.values)[0]
    )
    .to_frame(name="ret_wml_benchmark")
)

# SM signal-weighted
wml_sm_sw = (
    data.groupby("date")
    .apply(lambda x: np.sum(x["target"]*x["benchmark"])/np.sum(x["benchmark"].abs()))
    .to_frame(name="ret_sm_sw")
)

# CMM
wml = (
    ret_model.reset_index()
    .groupby("date")
    .apply(lambda x: (x[x["pf"] == 9].ret_model.values - x[x["pf"] == 0].ret_model.values)[0])
    .to_frame(name="ret_wml")
)

# CMP
wml_cmp = (
    data.groupby("date")
    .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
    .to_frame(name="ret_cmp")
)

# wml_ranks =  (  
#     data.groupby("date")
#     .apply(lambda x: np.sum(x["target"]*x["signal"])/np.sum(x["signal"].abs()) ) 
#     .to_frame(name="ret_ranks")
# )

# wml_lewellen =  (  
#     data.groupby("date")
#     .apply(lambda x: np.mean(x["target"]*(x["prediction"]- x["prediction"].mean()) ) )
#     .to_frame(name="ret_lewellen")
# )

# Barroso/Santa-Clara 2015
target = 12 / 100 / np.sqrt(12)     # NOTE Target volatility of 12% p.a.
sigma2 = pd.read_parquet("../03_data/wml_sigma2_forecast.pq")
ret_barroso = wml_benchmark["ret_wml_benchmark"].to_frame().copy()
ret_barroso = ret_barroso.merge(sigma2, on="date").set_index("date")
wml_barroso = (ret_barroso["ret_wml_benchmark"] * target/np.sqrt(ret_barroso["sigma2"])).to_frame(name="ret_barroso")

# CMP Excluding small stocks
wml_no_small = data[data.pf_size>=NSIZE_EXCLUDE]
wml_no_small = (
    wml_no_small.groupby("date")
    .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
    .to_frame(name="ret_no_small")
)

# ZERO COST CMP 
looser_cmp = data[data["prediction"]<0]
looser_cmp = (
    looser_cmp.groupby("date")
        .apply(lambda x: np.sum(x["target"]*x["prediction"].abs())/np.sum(x["prediction"].abs()))
        .to_frame(name="ret_looser")
    )

winner_cmp = data[data["prediction"]>=0]
winner_cmp = (
    winner_cmp.groupby("date")
        .apply(lambda x: np.sum(x["target"]*x["prediction"].abs())/np.sum(x["prediction"].abs()))
        .to_frame(name="ret_winner")
    )
wml_cmp_0 = (winner_cmp["ret_winner"] - looser_cmp["ret_looser"]).to_frame(name="ret_cmp_0")

# # ZERO COST Value-weighted
# looser = data[data["prediction"]<0]
# looser = (
# looser.groupby("date")
#     .apply(lambda x: np.sum(x["target"]*x["market_equity"])/np.sum(x["market_equity"]))
#     .to_frame(name="ret_looser")
# )

# winner = data[data["prediction"]>=0]
# winner = (
# winner.groupby("date")
#     .apply(lambda x: np.sum(x["target"]*x["market_equity"])/np.sum(x["market_equity"]))
#     .to_frame(name="ret_winner")
# )

# %%
###########################################################################
#
#               Double Sorts
#
###########################################################################

data["pf5_mom"] = data.groupby("date")["benchmark"].transform(lambda x: pd.qcut(x, 5, labels=False))
data["pf5_cmm"] = data.groupby("date")["prediction"].transform(lambda x: pd.qcut(x, 5, labels=False))


data["pf5_mom_cmm"] = data.groupby(["date", "pf5_mom"])["prediction"].transform(lambda x: pd.qcut(x, 5, labels=False))
data["pf5_cmm_mom"] = data.groupby(["date", "pf5_cmm"])["benchmark"].transform(lambda x: pd.qcut(x, 5, labels=False))


# -------------- Table -------------------
table_data = data.groupby(["date","pf5_mom", "pf5_cmm"])["target"].mean().to_frame()
pval_data = table_data.groupby(["pf5_mom", "pf5_cmm"])["target"].apply(
    lambda x: sm.OLS(exog=x, endog=np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).pvalues[0]
).unstack()

# HL for CMM
hl_cols = table_data.reset_index().groupby(["date","pf5_mom"]).apply(
    lambda x: x[x.pf5_cmm==4]["target"].values[0] - x[x.pf5_cmm==0]["target"].values[0]
).to_frame(name="r")
hl_cols_pval = hl_cols.groupby("pf5_mom")["r"].apply(
    lambda x: sm.OLS(exog=x.values, endog=np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).pvalues[0]
)

# HL for MOM
hl_rows = table_data.reset_index().groupby(["date", "pf5_cmm"]).apply(
    lambda x: x[x.pf5_mom==4]["target"].values[0] - x[x.pf5_mom==0]["target"].values[0]
).to_frame(name="r")
hl_rows_pval = hl_rows.groupby("pf5_cmm")["r"].apply(
    lambda x: sm.OLS(exog=x.values, endog=np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).pvalues[0]
)

# Calc mean and scale
table_data = table_data.groupby(["pf5_mom", "pf5_cmm"])["target"].mean().unstack() * 100
hl_cols = hl_cols.groupby(["pf5_mom"]).mean()* 100
hl_rows = hl_rows.groupby(["pf5_cmm"]).mean() * 100


# ------- Create Table
row_names = {
    0: r"\text{LOW}",
    1: r"\text{2}",
    2: r"\text{3}",
    3: r"\text{4}",
    4: r"\text{HIGH}"
}

col_names = {
    0: r"\text{LOW}",
    1: r"\text{2}",
    2: r"\text{3}",
    3: r"\text{4}",
    4: r"\text{HIGH}"
}


#table = Tabularx("l" + "".join(["c"] * (table_data.shape[1] + 1)), booktabs=True)
ncol = table_data.shape[1] + 1
table = Tabular("l" + "".join(["c"] * ncol), booktabs=True)
table.add_hline()
table.add_row(
    [   "",
        MultiColumn(5, align="c", data="CMM"),
        "",
    ]
)
table.add_hline(start=2, end=table_data.shape[1] + 1)
table.add_row(["SM"] + [math(col_names[k]) for k in col_names.keys()] + ["HL"])
table.add_hline()

for idx, row in table_data.iterrows():
    to_add = [math(row_names[idx])]

    for col, num in (row.iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            num_str = f"{num:.2f}"
            stars = ""
            if pval_data.loc[idx,col]<0.1:
                stars +="*"
            if pval_data.loc[idx,col]<0.05:
                stars +="*"
            if pval_data.loc[idx,col]<0.01:
                stars +="*"
            num_str = num_str +"^{" + stars + "}"
            to_add.append(math(align(num_str)))
    
    # ADD HL
    num = hl_cols.loc[idx,"r"]
    num_str = f"{num:.2f}"
    stars = ""
    if hl_cols_pval.loc[idx]<0.1:
        stars +="*"
    if hl_cols_pval.loc[idx]<0.05:
        stars +="*"
    if hl_cols_pval.loc[idx]<0.01:
        stars +="*"
    num_str = num_str +"^{" + stars + "}"
    to_add.append(math(align(num_str)))

    table.add_row(to_add)

# ADD HL
table.add_hline()
to_add = ["HL"]
for idx, num in hl_rows.iterrows():
    num_str = f"{num[0]:.2f}"
    stars = ""
    if hl_rows_pval.loc[idx]<0.1:
        stars +="*"
    if hl_rows_pval.loc[idx]<0.05:
        stars +="*"
    if hl_rows_pval.loc[idx]<0.01:
        stars +="*"
    num_str = num_str +"^{" + stars + "}"
    to_add.append(math(align(num_str)))
to_add.append("")
table.add_row(to_add)
    

# create .tex
table.add_hline()
table.generate_tex(os.path.join(saveLoc, f"ind_double_sort_ensemble={int(INT_ENSEMBLE)}"))

# %%
###########################################################################
#
#               Cumulative Return
#
###########################################################################


# ------------------------- Whole Sample ------------------------------------
start = "1980-02-01"
end = "2023-11-01"

plot_data = wml.loc[start:end].merge(wml_benchmark, on="date", how="left")
plot_data = plot_data.merge(wml_cmp,on="date", how="left")
plot_data = plot_data.merge(wml_barroso ,on="date", how="left")
plot_data = plot_data.merge(wml_no_small ,on="date", how="left")
plot_data = plot_data.merge(wml_cmp_0 ,on="date", how="left")
plot_data = plot_data.merge(ff, on="date", how="left")
min_date = plot_data.index.min()
# Add start of investment
plot_data = pd.concat([
    pd.DataFrame(0, columns=plot_data.columns, index=[min_date - pd.DateOffset(months=1)]),
    plot_data
])
plot_data = (plot_data +1 ).cumprod()

fig, ax = plt.subplots(figsize=(width, 7 * cm))
ax.plot(
    plot_data.index,
    plot_data.ret_wml_benchmark.values,
    label="SM",
    color=colors(2)[0])

# ax.plot(
#     plot_data.index,
#     plot_data.ret_wml.values,
#     label="CMM",
#     color=colors(2)[-1])

ax.plot(
    plot_data.index,
    plot_data["ret_cmp"].values,
    label="CMM",
    color=colors(2)[-1], 
    ls="-")

ax.plot(
    plot_data.index,
    plot_data["Mkt-RF"].values,
    label="MKT",
    color="k", 
    ls="--")


# ax.plot(
#     plot_data.index,
#     plot_data["ret_barroso"].values,
#     label="Barroso",
#     color="g", 
#     ls="--")

# ax.plot(
#     plot_data.index,
#     plot_data["ret_cmp_0"].values,
#     label="CMP0",
#     color="gray", 
#     ls="--")

# ax.plot(
#     plot_data.index,
#     plot_data["ret_no_small"].values,
#     label="Excluding Small",
#     color="gray", 
#     ls="--")



#ax.axhline(1, color="k", ls="--", lw=1.0)

ax.set_xlabel("Date")
ax.set_ylabel("Dollar value of investment")
ax.set_yscale("log")
ax.legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1)
ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax.set_xlim([plot_data.index.min(), plot_data.index.max()])


# gain = (wml.loc[start:end] + 1).prod().ret_wml / (wml_benchmark.loc[start:end] + 1).prod().ret_wml_benchmark
# m_model = (12 * wml.loc[start:end].mean() * 100).values[0]
# m_bench = (12 * wml_benchmark.loc[start:end].mean() * 100).values[0]
# sr_model = (np.sqrt(12) * wml.loc[start:end].mean()/wml.loc[start:end].std()).values[0]
# sr_bench = (np.sqrt(12) * wml_benchmark.loc[start:end].mean()/wml_benchmark.loc[start:end].std()).values[0]
# std_gain = wml.loc[start:end].std().values[0]/wml_benchmark.loc[start:end].std().values[0]
# ax.set_title(f"GAIN: {gain:.2f} -- SR: {sr_model:.3f} -- B-SR: {sr_bench:.3f} -- MEAN: {m_model:.1f}% -- B-MEAN: {m_bench:.1f}% ")

# -- Save
fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"cum_ret_full_sample_ensemble={int(INT_ENSEMBLE)}.pdf"), dpi=800)

# %%
# ------------------------- Whole Sample -- LOSER/WINNER ------------------------------------
start = "1980-02-01"
end = "2023-11-01"

plot_data = looser_cmp.loc[start:end].merge(winner_cmp, on="date", how="left")
plot_data = plot_data.merge(ff, on="date", how="left")
plot_data["ret_looser"] *= -1
min_date = plot_data.index.min()
# Add start of investment
plot_data = pd.concat([
    pd.DataFrame(0, columns=plot_data.columns, index=[min_date - pd.DateOffset(months=1)]),
    plot_data
])
plot_data = (plot_data +1 ).cumprod()

fig, ax = plt.subplots(figsize=(width, 7 * cm))
ax.plot(
    plot_data.index,
    plot_data.ret_looser.values,
    label="Short Leg",
    color=colors(2)[0])

ax.plot(
    plot_data.index,
    plot_data.ret_winner.values,
    label="Long Leg",
    color=colors(2)[-1])

ax.plot(
    plot_data.index,
    plot_data["Mkt-RF"].values,
    label="MKT",
    color="k", 
    ls="--")



ax.set_xlabel("Date")
ax.set_ylabel("Dollar value of investment")
ax.set_yscale("log")
ax.legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1)
ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)


# -- Save
fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"cum_ret_winner_looser_ensemble={int(INT_ENSEMBLE)}.pdf"), dpi=800)


# %%
# ---------------------------- Crashs ------------------------------
fig, ax = plt.subplots(1,2, figsize=(width, 6 * cm))

# Financial Crisis
start = "2008-02-01"
end = "2013-01-01"

plot_data = wml.loc[start:end].merge(wml_benchmark, on="date", how="left")
plot_data = plot_data.merge(wml_cmp,on="date", how="left")
plot_data = plot_data.merge(wml_barroso ,on="date", how="left")
plot_data = plot_data.merge(wml_no_small ,on="date", how="left")
plot_data = plot_data.merge(wml_cmp_0 ,on="date", how="left")
plot_data = plot_data.merge(ff, on="date", how="left")
min_date = plot_data.index.min()
# Add start of investment
plot_data = pd.concat([
    pd.DataFrame(0, columns=plot_data.columns, index=[min_date - pd.DateOffset(months=1)]),
    plot_data
])
plot_data = (plot_data +1 ).cumprod()


ax[0].plot(
    plot_data.index,
    plot_data.ret_wml_benchmark.values,
    label="SM",
    color=colors(2)[0])

# ax[0].plot(
#     plot_data.index,
#     plot_data.ret_wml.values,
#     label="CMM",
#     color=colors(2)[-1])

ax[0].plot(
    plot_data.index,
    plot_data["ret_cmp"].values,
    label="CMM",
    color=colors(2)[-1], 
    ls="-"
)

ax[0].plot(
    plot_data.index,
    plot_data["Mkt-RF"].values,
    label="MKT",
    color="k", 
    ls="--")



# ax[0].plot(
#     plot_data.index,
#     plot_data["ret_barroso"].values,
#     label="Barroso",
#     color="g", 
#     ls="--")


# ax[0].plot(
#     plot_data.index,
#     plot_data["ret_cmp_0"].values,
#     label="CMP0",
#     color="gray", 
#     ls="--")

# ax[0].plot(
#     plot_data.index,
#     plot_data["ret_no_small"].values,
#     label="Excluding Small",
#     color="gray", 
#     ls="--")


ax[0].set_xlabel("Date")
ax[0].set_ylabel("Dollar value of investment")
#ax[0].set_yscale("log")
#ax[0].legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1, ncol=2)
ax[0].grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax[0].set_xticks(pd.date_range(start[:4], end[:4], freq="1YS"))
ax[0].set_xlim([plot_data.index.min(), plot_data.index.max()])
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


# Corona
start = "2017-01-01"
end = "2022-01-01"

plot_data = wml.loc[start:end].merge(wml_benchmark, on="date", how="left")
plot_data = plot_data.merge(wml_cmp,on="date", how="left")
plot_data = plot_data.merge(wml_barroso ,on="date", how="left")
plot_data = plot_data.merge(wml_cmp_0 ,on="date", how="left")
#plot_data = plot_data.merge(wml_no_small ,on="date", how="left")
plot_data = plot_data.merge(ff, on="date", how="left")
min_date = plot_data.index.min()
# Add start of investment
plot_data = pd.concat([
    pd.DataFrame(0, columns=plot_data.columns, index=[min_date - pd.DateOffset(months=1)]),
    plot_data
])
plot_data = (plot_data +1 ).cumprod()


ax[1].plot(
    plot_data.index,
    plot_data.ret_wml_benchmark.values,
    label="SM",
    color=colors(2)[0])

# ax[1].plot(
#     plot_data.index,
#     plot_data.ret_wml.values,
#     label="CMM",
#     color=colors(2)[-1])

ax[1].plot(
    plot_data.index,
    plot_data["ret_cmp"].values,
    label="CMM",        # NOTE Wie changed the name for the Paper
    color=colors(2)[-1], 
    ls="-")

ax[1].plot(
    plot_data.index,
    plot_data["Mkt-RF"].values,
    label="MKT",
    color=colors(2)[-1], 
    ls="--")



# ax[1].plot(
#     plot_data.index,
#     plot_data["ret_barroso"].values,
#     label="Barroso",
#     color="g", 
#     ls="--")


# ax[1].plot(
#     plot_data.index,
#     plot_data["ret_cmp_0"].values,
#     label="CMP0",
#     color="gray", 
#     ls="--")


# ax[1].plot(
#     plot_data.index,
#     plot_data["ret_no_small"].values,
#     label="Excluding Small",
#     color="gray", 
#     ls="--")

ax[1].set_xlabel("Date")

#ax[1].set_yscale("log")
#ax[1].legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1, ncol=2)
ax[1].grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax[1].set_xticks(pd.date_range(start[:4], end[:4], freq="YS"))
#ax[1].set_xlim([start, end])
ax[1].set_xlim([plot_data.index.min(), pd.to_datetime(end)])
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ADD LEGEND
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.71, 1), ncol=3, fancybox=False, edgecolor="white")
#plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)

# -- Save
#fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"cum_ret_crisis_ensemble={int(INT_ENSEMBLE)}.pdf"), dpi=800, bbox_inches="tight")


# %%
###########################################################################
#
#               Performance Table
#
###########################################################################
def get_performance_info(wml, return_col: str):
    # Calculate Utility
    wml.loc[:,"u_gamma_5"] = crra(wml[return_col], gamma=5)
    wml.loc[:,"u_gamma_10"] = crra(wml[return_col], gamma=10)
    wml.loc[:,"u_gamma_100"] = crra(wml[return_col], gamma=100)

    wml_table = pd.DataFrame()
    # - Mean return
    wml_table.loc["r","All"] = wml[return_col].mean() * 12
    wml_table.loc["r","Pre 2003"] = wml[return_col][:"2002"].mean() * 12
    wml_table.loc["r","Post 2003"] = wml[return_col]["2003":].mean() * 12
    # - STD
    wml_table.loc["sigma", "All"] =  np.sqrt(12) * wml[return_col].std()
    wml_table.loc["sigma", "Pre 2003"] =  np.sqrt(12) * wml[:"2002"][return_col].std()
    wml_table.loc["sigma", "Post 2003"] =  np.sqrt(12) * wml["2003":][return_col].std()
    # - SR
    wml_table.loc["SR","All"] = wml_table.loc["r","All"]/wml_table.loc["sigma","All"]
    wml_table.loc["SR","Pre 2003"] = wml_table.loc["r","Pre 2003"]/wml_table.loc["sigma","Pre 2003"]
    wml_table.loc["SR","Post 2003"] = wml_table.loc["r","Post 2003"]/wml_table.loc["sigma","Post 2003"]
    # - MDD
    wml_table.loc["MDD", "All"] = calc_MDD((1+wml[return_col]).cumprod()).mdd.min()
    wml_table.loc["MDD", "Pre 2003"] = calc_MDD((1+wml[:"2002"][return_col]).cumprod()).mdd.min()
    wml_table.loc["MDD", "Post 2003"] = calc_MDD((1+wml["2003":][return_col]).cumprod()).mdd.min()
    # - SKEW
    wml_table.loc["Skew", "All"] = wml[return_col].skew()
    wml_table.loc["Skew", "Pre 2003"] = wml[:"2002"][return_col].skew()
    wml_table.loc["Skew", "Post 2003"] = wml["2003":][return_col].skew()
    # - KURTOSIS
    wml_table.loc["Kurtosis", "All"] = wml[return_col].kurt()
    wml_table.loc["Kurtosis", "Pre 2003"] = wml[:"2002"][return_col].kurt()
    wml_table.loc["Kurtosis", "Post 2003"] = wml["2003":][return_col].kurt()
    # - CE5
    wml_table.loc["CE5", "All"] = ce_crra(wml.loc[:,"u_gamma_5"].mean(), gamma=5) * 12
    wml_table.loc["CE5", "Pre 2003"] = ce_crra(wml.loc[:"2002","u_gamma_5"].mean(), gamma=5) * 12
    wml_table.loc["CE5", "Post 2003"] = ce_crra(wml.loc["2003":,"u_gamma_5"].mean(), gamma=5) * 12
    # - CE10
    wml_table.loc["CE10", "All"] = ce_crra(wml.loc[:,"u_gamma_5"].mean(), gamma=10) * 12
    wml_table.loc["CE10", "Pre 2003"] = ce_crra(wml.loc[:"2002","u_gamma_10"].mean(), gamma=10) * 12
    wml_table.loc["CE10", "Post 2003"] = ce_crra(wml.loc["2003":,"u_gamma_10"].mean(), gamma=10) * 12
    # - CE100
    wml_table.loc["CE100", "All"] = ce_crra(wml.loc[:,"u_gamma_100"].mean(), gamma=100) * 12
    wml_table.loc["CE100", "Pre 2003"] = ce_crra(wml.loc[:"2002","u_gamma_100"].mean(), gamma=100) * 12
    wml_table.loc["CE100", "Post 2003"] = ce_crra(wml.loc["2003":,"u_gamma_100"].mean(), gamma=100) * 12

    return wml_table

# --- MODEL
wml_table = get_performance_info(wml, return_col="ret_wml")

# --- MODEL - CMP
cmp_table = get_performance_info(wml_cmp, return_col="ret_cmp")

# --- MODEL - CMP0
cmp_0_table = get_performance_info(wml_cmp_0, return_col="ret_cmp_0")

# --- MODEL - CMP
cmp_ex_table = get_performance_info(wml_no_small, return_col="ret_no_small")

# --- Barroso
barroso_table = get_performance_info(wml_barroso, return_col="ret_barroso")

# ---  Benchmark
benchmark_table = get_performance_info(wml_benchmark, return_col="ret_wml_benchmark")


# ------- Create Table
row_names = {
    "r": r"$\overline{r}$",
    "sigma": r"$\sigma(r)$",
    "SR": r"\text{SR}",
    "MDD": r"\text{MDD}",
    "Skew": r"\text{Skewness}",
    "Kurtosis": r"\text{Kurtosis}",
    "CE5": r"\text{CE}($\gamma=5$)",
    #"CE10": r"\text{CE}($\gamma=10$)",
    #"CE100": r"\text{CE}($\gamma=100$)",
}
table = Tabularx("l" + "".join(["c"] * (2 * wml_table.shape[1] + 1)), booktabs=True)
table.add_hline()
table.add_row(
    [
        "",
        MultiColumn(3, align="c", data="CMM"),
        "",
        MultiColumn(3, align="c", data="SM"),
    ]
)
table.add_hline(start=2, end=4)
table.add_hline(start=6, end=8)
table.add_row([""] + list(wml_table.columns) + [""] + list(benchmark_table.columns))
table.add_hline()

for r1, r2 in zip(cmp_table.iterrows(),benchmark_table.iterrows()):
    if r1[0] not in row_names.keys():
        continue

    to_add = [math(row_names[r1[0]])]
    # --- CMM
    for col, num in (r1[1].iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.3f}"))
    
    to_add.append("")
    # --- SM
    for col, num in (r2[1].iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.3f}"))
    table.add_row(to_add)

table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"overview_table_ensemble={int(INT_ENSEMBLE)}"))

# %%
###########################################################################
#
#          Benchmarking
#
###########################################################################

# Ret_12_7 (Novy-Marx (2012)): Decile Spreads based on ret_12_7, value weighted
data["pf_ret_12_7"] = data.groupby("date")["ret_12_7"].transform(lambda x: pd.qcut(x, q=10, labels=False))
ret_12_7 = data.groupby(["date", "pf_ret_12_7"]).apply(lambda x: vw(x["target"], x["market_equity"])).to_frame("ret_pf_12_7")
wml_12_7 = (
    ret_12_7.reset_index().groupby("date")
    .apply(
        lambda x: (x[x["pf_ret_12_7"] == 9].ret_pf_12_7.values - x[x["pf_ret_12_7"] == 0].ret_pf_12_7.values)[0]
    ).to_frame(name="ret_wml_12_7")
)

# Ret_6_1 (Novy-Marx (2012)): Decile Spreads based on ret_6_1, value weighted
data["pf_ret_6_1"] = data.groupby("date")["ret_6_1"].transform(lambda x: pd.qcut(x, q=10, labels=False))
ret_6_1 = data.groupby(["date", "pf_ret_6_1"]).apply(lambda x: vw(x["target"], x["market_equity"])).to_frame("ret_pf_6_1")
wml_6_1 = (
    ret_6_1.reset_index().groupby("date")
    .apply(
        lambda x: (x[x["pf_ret_6_1"] == 9].ret_pf_6_1.values - x[x["pf_ret_6_1"] == 0].ret_pf_6_1.values)[0]
    ).to_frame(name="ret_wml_6_1")
)
# prc_highprc_252d (George and Hwan (2004)): "The winner (loser) portfolio for the 52week high strategy 
# is the equally weighted portfolio of the 30% of stocks with the highest (lowest) 
# ratio of current price to 52-week high."
data["pf_prc_highprc"] = data.groupby("date")["prc_highprc_252d"].transform(lambda x: pd.qcut(x, q=[0, 0.3 ,0.7, 1], labels=False))
ret_prc_highprc = data.groupby(["date", "pf_prc_highprc"])["target"].mean().to_frame("ret_pf_prc_highprc")
wml_prc_highprc = (
    ret_prc_highprc.reset_index().groupby("date")
    .apply(
        lambda x: (x[x["pf_prc_highprc"] == 2].ret_pf_prc_highprc.values - x[x["pf_prc_highprc"] == 0].ret_pf_prc_highprc.values)[0]
    ).to_frame(name="ret_prc_highprc")
)

# Barroso/Santa-Clara 2015
sigma2 = pd.read_parquet("../03_data/wml_sigma2_forecast.pq")
ret_barroso = wml_benchmark["ret_wml_benchmark"].to_frame().copy()
ret_barroso = ret_barroso.merge(sigma2, on="date").set_index("date")
wml_barroso = (ret_barroso["ret_wml_benchmark"] * barroso_target/np.sqrt(ret_barroso["sigma2"])).to_frame(name="ret_barroso")


def calc_benchmark_info(wml, delta_w, turnover, return_col:str, wml_benchmark=wml_benchmark):
    # get col of benchmark
    bench_col = [c for c in wml_benchmark.columns if "ret_wml" in c][0]
    # Transaction costs
    transaction_costs = (delta_w.abs() * 25 / 100 / 100).sum(axis=1)

    # --- Merge data
    tmp = wml[return_col].to_frame().merge(transaction_costs.to_frame(name="t_cost"), on="date")
    tmp = tmp.merge(turnover.to_frame(name="turn"), on="date")

    # Max trading costs to be on par with benchmark
    if np.sum(wml.index != wml_benchmark.index)>0:
        ValueError("Check calc_benchmark_info function: indices do not match.")
    
    #NOTE: previous estimate, now: Equal SR and per Trad
    # ret_diff = (wml[return_col] - wml_benchmark[bench_col]).to_frame(name="ret_diff")
    # ret_diff["sum_delta_w"] = delta_w.abs().sum(axis=1)
    # max_costs = ret_diff["ret_diff"]/ret_diff["sum_delta_w"]

    # TC to have same SR
    if bench_col not in tmp.columns:
        tmp = tmp.merge(wml_benchmark[bench_col], on="date")
    tcpar = tmp[return_col] - (tmp[bench_col]*tmp[return_col].std()/tmp[bench_col].std())
    tcpar_per_trade = tcpar.mean() / (delta_w.abs().sum(axis=1)).mean()

    # Net return
    tmp.loc[:,"r_net"] = tmp[return_col] - tmp["t_cost"]

    # Create table
    out = pd.Series(dtype="float64")
    out.loc["r"] = tmp[return_col].mean()*12
    out.loc["turnover"] = tmp["turn"].mean()
    out.loc["transaction_cost"] = tmp["t_cost"].mean() * 12 * 100
    out.loc["transaction_costs_par_sm"] = tcpar_per_trade * 100
    out.loc["r_net"] = tmp["r_net"].mean()*12
    out.loc["sigma"] = np.sqrt(12) * tmp[return_col].std()
    out.loc["sigma_net"] = np.sqrt(12) * tmp["r_net"].std()
    out.loc["SR"] = out["r"]/out["sigma"]
    out.loc["SR_net"] = out["r_net"]/out["sigma_net"]

    return out

final_benchmark = pd.DataFrame()

# -------------------------------- Our model - CMP --------------------------------
# Normalize weights and calculate turnover
cmp = data[["prediction"]].copy()
cmp = cmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = cmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
# NOTE Changed to non scaled
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_cmp, delta_w, turnover, return_col="ret_cmp")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="cmp")], axis=1)

# -------------------------------- Our model - CMP ZERO COST--------------------------------
tmp = data[["prediction", "pf_prediction"]].copy()
tmp = tmp.reset_index().groupby(["date", "pf_prediction"]).apply(norm_func, norm_col="prediction")
tmp = tmp.drop(columns=["pf_prediction"])
admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_cmp_0, delta_w, turnover, return_col="ret_cmp_0")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="CMP0")], axis=1)

# -------------------------------- Our model --------------------------------
cmm = data[["pf", "market_equity"]].copy()
cmm = cmm.reset_index().groupby(["date", "pf"]).apply(norm_func)
cmm = cmm[cmm.pf.isin([cmm.pf.min(), cmm.pf.max()])]
cmm.loc[cmm.pf==cmm.pf.min(),"w"] *= -1
cmm = cmm.drop(columns=["pf"])

# Turnover
admissable_weights = cmm.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml, delta_w, turnover, return_col="ret_wml")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="cmm")], axis=1)


# -------------------------------- Normal Momentum   --------------------------------
# Normalize weights and calculate turnover
tmp = data[["pf_benchmark", "market_equity"]].copy()
tmp = tmp.reset_index().groupby(["date", "pf_benchmark"]).apply(norm_func)
tmp = tmp[tmp.pf_benchmark.isin([tmp.pf_benchmark.min(), tmp.pf_benchmark.max()])]
tmp.loc[tmp.pf_benchmark==tmp.pf_benchmark.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_benchmark"])

admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_benchmark, delta_w, turnover, return_col="ret_wml_benchmark")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="ret_12_1")], axis=1)

# -------------------------------- Normal Momentum signal weighted --------------------------------
# Normalize weights and calculate turnover
tmp = data[["benchmark"]].copy()
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="benchmark")
admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_sm_sw, delta_w, turnover, return_col="ret_sm_sw")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="ret_12_1_SW")], axis=1)

# -------------------------------- Ret_12_7   --------------------------------
# Normalize weights and calculate turnover
tmp = data[["pf_ret_12_7", "market_equity"]].copy()
tmp = tmp.reset_index().groupby(["date", "pf_ret_12_7"]).apply(norm_func)
tmp = tmp[tmp.pf_ret_12_7.isin([tmp.pf_ret_12_7.min(), tmp.pf_ret_12_7.max()])]
tmp.loc[tmp.pf_ret_12_7==tmp.pf_ret_12_7.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_ret_12_7"])

admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_12_7, delta_w, turnover, return_col="ret_wml_12_7")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="ret_12_7")], axis=1)


# -------------------------------- Ret_6_1   --------------------------------
# Normalize weights and calculate turnover
tmp = data[["pf_ret_6_1", "market_equity"]].copy()
tmp = tmp.reset_index().groupby(["date", "pf_ret_6_1"]).apply(norm_func)
tmp = tmp[tmp.pf_ret_6_1.isin([tmp.pf_ret_6_1.min(), tmp.pf_ret_6_1.max()])]
tmp.loc[tmp.pf_ret_6_1==tmp.pf_ret_6_1.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_ret_6_1"])

admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_6_1, delta_w, turnover, return_col="ret_wml_6_1")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="ret_6_1")], axis=1)


# -------------------------------- prc_highprc_252d  --------------------------------
# Normalize weights and calculate turnover
tmp = data[["pf_prc_highprc", "market_equity"]].copy()
tmp = tmp.reset_index().groupby(["date", "pf_prc_highprc"]).apply(norm_func, equal_weights=True)
tmp = tmp[tmp.pf_prc_highprc.isin([tmp.pf_prc_highprc.min(), tmp.pf_prc_highprc.max()])]
tmp.loc[tmp.pf_prc_highprc==tmp.pf_prc_highprc.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_prc_highprc", "market_equity"])

admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_prc_highprc, delta_w, turnover, return_col="ret_prc_highprc")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="prc_highprc")], axis=1)


# -------------------------------- Barroso/Santa-Clara 2015 --------------------------------
# NOTE Similar to SM but weights do not sum to 1 in long and short leg
tmp = data[["pf_benchmark", "market_equity"]] 
tmp = tmp.reset_index().groupby(["date", "pf_benchmark"]).apply(norm_func)
tmp = tmp[tmp.pf_benchmark.isin([tmp.pf_benchmark.min(), tmp.pf_benchmark.max()])]
tmp.loc[tmp.pf_benchmark==tmp.pf_benchmark.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_benchmark"])
tmp = tmp.merge(sigma2, on="date")
tmp["share_invested"] = barroso_target/np.sqrt(tmp["sigma2"])
tmp["w"] *= tmp["share_invested"]
tmp = tmp.drop(columns=["sigma2", "share_invested"]) 

admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
#turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = admissable_weights.diff().abs().sum(axis=1) 
turnover = turnover.replace([-np.inf, np.inf], np.nan)

# Transaction costs
delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

tmp = calc_benchmark_info(wml_barroso, delta_w, turnover, return_col="ret_barroso")
final_benchmark = pd.concat([final_benchmark, tmp.to_frame(name="barroso")], axis=1)

# Set transaction_cost_par_sm for momentum to 0
final_benchmark.loc["transaction_costs_par_sm","ret_12_1"] = ""
# %%
# ------- Create Table
row_names = {
    "r": r"$\overline{r}$",
    "sigma": r"$\sigma(r)$",
    "SR": r"\text{SR}",
    "SR_net": r"$\text{SR}^\text{net}$",
    "turnover": r"\text{TURN}",
    "transaction_cost": r"\text{TC} \ [\%]",
    "transaction_costs_par_sm": r"$\text{TC}^\text{par}$ [\%]",
}
cols_to_use = {
    "cmp": r"\text{CMM}",
    "ret_12_1": r"\text{SM}",
    "ret_12_1_SW": r"$\text{SM}^\text{sw}$",
    "ret_12_7": r"\text{MOM\_12\_7}",
    "ret_6_1": r"\text{MOM\_6\_1}",
    "prc_highprc": r"\text{GH04}",
    "barroso": r"\text{BSC16}",
}
table_data = final_benchmark.loc[row_names.keys(),cols_to_use.keys()]

table = Tabularx("l" + "".join(["c"] * (table_data.shape[1])), booktabs=True)
table.add_hline()
table.add_row([""] + [math(col) for _,col in cols_to_use.items()])
table.add_hline()

for idx, row in table_data.iterrows():
    to_add = [math(row_names[idx])]

    for col, num in (row.iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.3f}"))
    table.add_row(to_add)

table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"benchmark_table_ensemble={int(INT_ENSEMBLE)}"))



# %%
###########################################################################
#
#          Benchmarking LONG LEG
#
###########################################################################

long_benchmark = pd.DataFrame()

# -------------------------------- Our model --------------------------------
# Normalize weights and calculate turnover
tmp = data[["prediction"]].copy()
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

# ALL
tmp = calc_benchmark_info(wml_cmp, delta_w, turnover, return_col="ret_cmp")
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="cmp_all")], axis=1)
# PRE
tmp = calc_benchmark_info(wml_cmp[:"2002":], delta_w[:"2002"], turnover[:"2002"], return_col="ret_cmp", wml_benchmark=wml_benchmark[:"2002"])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="cmp_pre")], axis=1)
# POST
tmp = calc_benchmark_info(wml_cmp["2003":], delta_w["2003":], turnover["2003":], return_col="ret_cmp",wml_benchmark=wml_benchmark["2003":])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="cmp_post")], axis=1)



# -------------------------------- Our model - LONG --------------------------------
# Normalize weights and calculate turnover
tmp = data[data["prediction"]>=0][["prediction"]].copy()
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

# ALL
tmp = calc_benchmark_info(winner_cmp, delta_w, turnover, return_col="ret_winner")
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="winner_all")], axis=1)
# PRE
tmp = calc_benchmark_info(winner_cmp[:"2002":], delta_w[:"2002"], turnover[:"2002"], return_col="ret_winner", wml_benchmark=wml_benchmark[:"2002"])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="winner_pre")], axis=1)
# POST
tmp = calc_benchmark_info(winner_cmp["2003":], delta_w["2003":], turnover["2003":], return_col="ret_winner",wml_benchmark=wml_benchmark["2003":])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="winner_post")], axis=1)



# -------------------------------- Our model - SHORT --------------------------------
# Normalize weights and calculate turnover
tmp = data[data["prediction"]<0][["prediction"]].copy()
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = tmp.set_index(["date", "permno"]).unstack().fillna(0)
turnover = admissable_weights.diff().abs().sum(axis=1) / admissable_weights.abs().sum(axis=1)
turnover = turnover.replace([-np.inf, np.inf], np.nan)

delta_w = admissable_weights.diff()
delta_w.iloc[0] = admissable_weights.iloc[0]

# ALL
tmp = calc_benchmark_info(-looser_cmp, delta_w, turnover, return_col="ret_looser")
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="looser_all")], axis=1)
# PRE
tmp = calc_benchmark_info(-looser_cmp[:"2002":], delta_w[:"2002"], turnover[:"2002"], return_col="ret_looser", wml_benchmark=wml_benchmark[:"2002"])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="looser_pre")], axis=1)
# POST
tmp = calc_benchmark_info(-looser_cmp["2003":], delta_w["2003":], turnover["2003":], return_col="ret_looser",wml_benchmark=wml_benchmark["2003":])
long_benchmark = pd.concat([long_benchmark, tmp.to_frame(name="looser_post")], axis=1)



# %%
# ------- Create Table
row_names = {
    "r": r"$\overline{r}$",
    "sigma": r"$\sigma(r)$",
    "SR": r"\text{SR}",
    "SR_net": r"$\text{SR}^\text{net}$",
    "turnover": r"\text{TURN}",
    "transaction_cost": r"\text{TC} \ [\%]",
    "transaction_costs_par_sm": r"$\text{TC}^\text{par}$ [\%]",
}

table_data = long_benchmark.loc[row_names.keys(),:]

table = Tabularx("l" + "".join(["c"] * (2*3+1)), booktabs=True)
table.add_hline()
table.add_row(
    [
        "",
        MultiColumn(3, align="c", data="CMM"),
        "",
        MultiColumn(3, align="c", data=math(r"$\text{CMM}^\text{LONG}$")),
    ]
)
table.add_hline(start=2, end=4)
table.add_hline(start=6, end=8)
table.add_row([""] + ["All", "Pre 2003", "Post 2003"] + [""] + ["All", "Pre 2003", "Post 2003"])
table.add_hline()

for idx, row in table_data.iterrows():
    to_add = [math(row_names[idx])]

    # CMM
    to_use = ["cmp_all", "cmp_pre", "cmp_post"]
    for col, num in (row[to_use].iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.3f}"))

    to_add.append("")
    # WINNER
    to_use = ["winner_all", "winner_pre", "winner_post"]
    for col, num in (row[to_use].iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.3f}"))
    table.add_row(to_add)

table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"benchmark_long_ensemble={int(INT_ENSEMBLE)}"))


# %%
###########################################################################
#
#          Weight per Size Portfolio
#
###########################################################################
final_size_weights = pd.DataFrame()

# --- OUR MODEL CMP
tmp = data[["prediction", "pf_size"]]
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = tmp.set_index(["date", "permno"]).fillna(0)
weights_size = admissable_weights.groupby(["date","pf_size"])["w"].apply(lambda x: np.sum(x.abs()))
weights_size = weights_size.groupby("pf_size").mean().to_frame(name="CMP")
final_size_weights = pd.concat([final_size_weights, weights_size.T], axis=0)

# --- OUR MODEL CMP EXCLUDING SMALL
tmp = data[data.pf_size >=NSIZE_EXCLUDE][["prediction", "pf_size"]]
tmp = tmp.reset_index().groupby(["date"]).apply(norm_func, norm_col="prediction")
admissable_weights = tmp.set_index(["date", "permno"]).fillna(0)
weights_size = admissable_weights.groupby(["date","pf_size"])["w"].apply(lambda x: np.sum(x.abs()))
weights_size = weights_size.groupby("pf_size").mean().to_frame(name="CMP EX ")
final_size_weights = pd.concat([final_size_weights, weights_size.T], axis=0)

# --- OUR MODEL CMP ZERO COST
tmp = data[["prediction", "pf_prediction", "pf_size"]]
tmp = tmp.reset_index().groupby(["date", "pf_prediction"]).apply(norm_func, norm_col="prediction")
tmp = tmp.drop(columns=["pf_prediction"])
admissable_weights = tmp.set_index(["date", "permno"]).fillna(0)
weights_size = admissable_weights.groupby(["date","pf_size"])["w"].apply(lambda x: np.sum(x.abs())/2)
weights_size = weights_size.groupby("pf_size").mean().to_frame(name="CMP0")
final_size_weights = pd.concat([final_size_weights, weights_size.T], axis=0)

# --- OUR MODEL CMM
tmp = data[["pf", "market_equity", "pf_size"]]
tmp = tmp.reset_index().groupby(["date", "pf"]).apply(norm_func)
tmp = tmp[tmp.pf.isin([tmp.pf.min(), tmp.pf.max()])]
tmp.loc[tmp.pf==tmp.pf.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf"])

admissable_weights = tmp.set_index(["date", "permno"]).fillna(0)
weights_size = admissable_weights.groupby(["date","pf_size"])["w"].apply(lambda x: np.sum(x.abs())/2)
weights_size = weights_size.groupby("pf_size").mean().to_frame(name="CMM")
final_size_weights = pd.concat([final_size_weights, weights_size.T], axis=0)

# --- Normal Momentum
tmp = data[["pf_benchmark", "market_equity", "pf_size"]]
tmp = tmp.reset_index().groupby(["date", "pf_benchmark"]).apply(norm_func)
tmp = tmp[tmp.pf_benchmark.isin([tmp.pf_benchmark.min(), tmp.pf_benchmark.max()])]
tmp.loc[tmp.pf_benchmark==tmp.pf_benchmark.min(),"w"] *= -1
tmp = tmp.drop(columns=["pf_benchmark"])

admissable_weights = tmp.set_index(["date", "permno"]).fillna(0)
weights_size = admissable_weights.groupby(["date","pf_size"])["w"].apply(lambda x: np.sum(x.abs())/2)
weights_size = weights_size.groupby("pf_size").mean().to_frame(name="SM")
final_size_weights = pd.concat([final_size_weights, weights_size.T], axis=0)

final_size_weights = final_size_weights.fillna(0)


# %%
###########################################################################
#
#               sharpe ratio excluding small
#
###########################################################################


fn = os.path.join(saveLoc, f"sr_per_size_ensemble={int(INT_ENSEMBLE)}.pq")
if os.path.exists(fn):
    sr_per_size = pd.read_parquet(fn)
else:
    sr_per_size = pd.DataFrame()
    for lim in np.arange(0,91):
        print(f"\rWorking on SR for SIZE-LIM: {int(lim)}", flush=True, end="")

        out = pd.DataFrame(index=[lim])
        out.index.name = "cutoff"
        
        # --- OUR MODEL CMP
        tmp = data[data.size_percentile>=lim].copy()
        sr = (
            tmp.groupby("date")
            .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
            .to_frame(name="ret")
        )
        #out["mean_cmp"] = sr.mean()[0] * 12
        #out["std_cmp"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_cmp"] = sr

        # --- OUR MODEL CMP POST 2003
        tmp = data["2003":].copy()
        tmp = tmp[tmp.size_percentile>=lim]
        sr = (
            tmp.groupby("date")
            .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
            .to_frame(name="ret")
        )
        #out["mean_cmp"] = sr.mean()[0] * 12
        #out["std_cmp"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_cmp_post"] = sr

        # --- OUR MODEL CMP LONG ONLY
        tmp = data[data.size_percentile>=lim].copy()
        tmp = tmp[tmp["prediction"]>=0]
        sr = (
            tmp.groupby("date")
            .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
            .to_frame(name="ret")
        )
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_cmp_long"] = sr

        # --- OUR MODEL CMP LONG ONLY Post 2003
        tmp = data["2003":].copy()
        tmp = tmp[tmp.size_percentile>=lim]
        tmp = tmp[tmp["prediction"]>=0]
        sr = (
            tmp.groupby("date")
            .apply(lambda x: np.sum(x["target"]*x["prediction"])/np.sum(x["prediction"].abs()))
            .to_frame(name="ret")
        )
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_cmp_long_post"] = sr

        # ---- OUR MODEL CMM
        tmp = data[data.size_percentile>=lim].copy()
        tmp.loc[:,"pf"] = tmp.groupby("date")["prediction"].transform(lambda x: pd.qcut(x, q=10, labels=False))
        sr = (
            tmp.groupby(["date", "pf"])
            .apply(lambda x: vw(x["target"], x["market_equity"]))
            .to_frame("ret_cmm")
        )
        sr = (
            sr.reset_index()
            .groupby("date")
            .apply(
                lambda x: (x[x["pf"] == 9].ret_cmm.values - x[x["pf"] == 0].ret_cmm.values)[0] 
            )
            .to_frame(name="ret")
        )
        #out["mean_cmm"] = sr.mean()[0] * 12
        #out["std_cmm"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_cmm"] = sr 

        # --- STANDARD MOMENTUM - signal weighted
        tmp = data[data.size_percentile>=lim].copy()
        sr = (
            tmp.groupby("date")
            .apply(lambda x: np.sum(x["target"]*x["benchmark"])/np.sum(x["benchmark"].abs()))
            .to_frame(name="ret")
        )
        #out["mean_sm_sw"] = sr.mean()[0] * 12
        #out["std_sm_sw"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_sm_sw"] = sr 
            

        # --- STANDARD MOMENTUM
        tmp = data[data.size_percentile>=lim].copy()
        tmp.loc[:,"pf_benchmark"] = tmp.groupby("date")["benchmark"].transform(lambda x: pd.qcut(x, q=10, labels=False))
        sr = (
            tmp.groupby(["date", "pf_benchmark"])
            .apply(lambda x: vw(x["target"], x["market_equity"]))
            .to_frame("ret_benchmark")
        )
        sr = (
            sr.reset_index()
            .groupby("date")
            .apply(
                lambda x: (x[x["pf_benchmark"] == 9].ret_benchmark.values - x[x["pf_benchmark"] == 0].ret_benchmark.values)[0]
            )
            .to_frame(name="ret")
        )
        #out["mean_sm"] = sr.mean()[0] * 12
        #out["std_sm"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_sm"] = sr 

        # --- STANDARD MOMENTUM POST 2003
        tmp = data[data.size_percentile>=lim].copy()
        tmp = tmp["2003":]
        tmp.loc[:,"pf_benchmark"] = tmp.groupby("date")["benchmark"].transform(lambda x: pd.qcut(x, q=10, labels=False))
        sr = (
            tmp.groupby(["date", "pf_benchmark"])
            .apply(lambda x: vw(x["target"], x["market_equity"]))
            .to_frame("ret_benchmark")
        )
        sr = (
            sr.reset_index()
            .groupby("date")
            .apply(
                lambda x: (x[x["pf_benchmark"] == 9].ret_benchmark.values - x[x["pf_benchmark"] == 0].ret_benchmark.values)[0]
            )
            .to_frame(name="ret")
        )
        #out["mean_sm"] = sr.mean()[0] * 12
        #out["std_sm"] = sr.std()[0] * np.sqrt(12)
        sr = np.sqrt(12) * (sr.mean().values / sr.std().values)[0]
        out["sr_sm_post"] = sr 

        sr_per_size = pd.concat([sr_per_size, out])

    sr_per_size.to_parquet(fn)   

# %%
# ------ Sharpe Ratio
fig, ax = plt.subplots(figsize=(width, 6 * cm))
# NOTE plot every second may look better
ax.plot(
    sr_per_size.iloc[::2].index.values,
    sr_per_size.iloc[::2]["sr_cmp"].values,
    color = colors(2)[-1],
    label="CMM",
    ls="--",
    marker="o"
    )
ax.plot(
    sr_per_size.iloc[::2].index.values,
    sr_per_size.iloc[::2]["sr_cmp_long"].values,
    color = colors(2)[-1],
    label=r"$\mathrm{CMM}^\mathrm{LONG}$",
    ls="--",
    marker="*"
    )

ax.plot(
    sr_per_size.iloc[::2].index.values,
    sr_per_size.iloc[::2]["sr_sm"].values,
    color = colors(2)[0],
    label="SM",
    ls="--",
    marker="o"
    )


ax.legend(fancybox=False, edgecolor="white", loc="upper right", framealpha=1)
ax.set_xlim([0,80])
ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
#ax.set_ylim([0,1])
ax.set_xlabel("Size Cutoff Percentile")
ax.set_ylabel("Sharpe Ratio (SR)")

fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"sr_by_size_cutoff_ensemble={int(INT_ENSEMBLE)}.pdf"), dpi=800)


# %%
###########################################################################
#
#          Get Attention Data
#
###########################################################################
# Concat results
att = Parallel(os.cpu_count()//2, backend="loky", verbose=True)(
    delayed(reader)(fn, attention=True, ENSEMBLE=ENSEMBLE, N=N_ENSEMBLE) for fn in folder_list

)
att = pd.concat(att)
att = att.reset_index(drop=True).sort_values("date").set_index(["date", "id"])




# %%
###########################################################################
#
#               Sparsity in Weights
#
###########################################################################
quants = [0.25, 0.75]
sorted_att = pd.DataFrame(
    np.cumsum(np.sort(att), axis=1),
    index=att.index,
    columns=np.arange(att.shape[1])
)

disp_curve = sorted_att.mean().to_frame(name="mean")
disp_curve["qlow"] =sorted_att.quantile(quants[0])
disp_curve["qhigh"] =sorted_att.quantile(quants[1])

# ---- Plot
N = disp_curve.index.max()

fig, ax = plt.subplots(figsize=(width, 6 * cm))
ax.plot(
    disp_curve.index/N*100,
    disp_curve["mean"].values,
    color = colors(3)[2],
    label="Mean"
)
ax.fill_between(
    disp_curve.index/N*100,
    disp_curve.qlow,
    disp_curve.qhigh,
    color=colors(3)[0],
    alpha=0.5,
    label="[Q25, Q75]"
)
ax.plot(
    disp_curve.index/N*100,
    np.linspace(0,1,disp_curve.shape[0]),
    color="k",
    ls="--",
    label="Equal Importance"
)
ax.legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1)
ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax.set_xlim([0,100])
ax.set_ylim([0,1])
ax.set_xlabel("% of look-back window")
ax.set_ylabel("Cumulative weight")

fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"disparity_curve_ensemble={int(INT_ENSEMBLE)}.pdf"), dpi=800)

print("--------------------------- Sparsity ------------------------")
for i in [1, 5, 10, 15, 20]:
    tmp = (1-disp_curve.iloc[np.argmax((disp_curve.index/N*100)>=(100-i))]["mean"]) *100
    print(f"{i}% of all observations account for {tmp:.2f}% of the weights.")

print("-------------------------------------------------------------")


# %%
###########################################################################
#
#               Calculate Weight in EA/FOMC/...
#
###########################################################################

# ---- Get trading dates
fn = "../03_data/crsp_trading_dates.pq"
if os.path.exists(fn):
    trading_dates = pd.read_parquet(fn)
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
                            a.date >= '01/01/1960'
        """
    )

    # ---- specify format and save
    crsp["date"] = pd.to_datetime(crsp["date"], format="%Y-%m-%d")
    crsp = crsp.astype({"permno": "i4"})
    crsp = crsp.sort_values("date").reset_index(drop=True)
    crsp.to_parquet(fn)

    # --- Close db
    db.close()


fn = os.path.join(saveLoc,f"weight_analysis_ensemble={int(INT_ENSEMBLE)}.pq")
if os.path.exists(fn):
    weights = pd.read_parquet(os.path.join(saveLoc,f"weight_analysis_ensemble={int(INT_ENSEMBLE)}.pq"))
else:
    trading_grid = pd.Series(trading_dates["date"].unique())
    weights = Parallel(n_jobs=CORES, verbose=True)(
        delayed(agg_weights)(
            w,
            trading_grid,
            window=2,
            ea = ea[ea.permno == idx[1]],
            gdp = gdp, cpi=cpi, fomc=fomc, 
            r = trading_dates[trading_dates.permno == idx[1]],
            mkt = mkt_daily
        ) for idx, w in att.iterrows()
    ) # NOTE About 2.4 mio firm-month obs
    weights = pd.concat(weights).reset_index(drop=True)
    weights.to_parquet(fn)


# %%
###########################################################################
#
#               Plot weights
#
###########################################################################

def get_rel_importance(df, name):

    # NOTE Change in future version as we miss 1 day..
    n_12_7 = 126
    n_6_1 = 104
        
    df = df.dropna()
    if name in ["12_7", "6_1"]:
        norm = n_12_7 if name=="12_7" else n_6_1
        equal_w = 1/(n_12_7 + n_6_1)
        name_w = (df[f"w_{name}"]/norm) 
    else:
        # NOTE Weights seem to somtetimes not sum to 231. Check code. Until now, I drop these days
        df = df[(df[f"n_w_{name}"] + df[f"n_w_non_{name}"])== 231] 
        equal_w = 1/(df[f"n_w_{name}"] + df[f"n_w_non_{name}"])
        name_w = (df[f"w_{name}"]/df[f"n_w_{name}"]) 
    
    w = name_w/equal_w
    avg = pd.DataFrame({
        "w_mean": np.nanmean(w),
        "w_q10": np.quantile(w, 0.1),
        "w_q25": np.quantile(w, 0.25),
        "w_q50": np.quantile(w, 0.5),
        "w_q75": np.quantile(w, 0.75),
        "w_q90": np.quantile(w, 0.90),
        "w_q99": np.quantile(w, 0.99),
        "w_q999": np.quantile(w, 0.999),
        "w_std": np.nanstd(w),
        "type": name.upper()
    }, index=[0])

    return avg

if "date" in weights.columns:
    weights = weights.set_index("date")
# ALL
plot_data = []
plot_data.append(get_rel_importance(weights[["w_ea", "n_w_ea", "n_w_non_ea"]], name="ea"))
plot_data.append(get_rel_importance(weights[["w_fomc", "n_w_fomc", "n_w_non_fomc"]], name="fomc"))
plot_data.append(get_rel_importance(weights[["w_cpi", "n_w_cpi", "n_w_non_cpi"]], name="cpi"))
plot_data.append(get_rel_importance(weights[["w_baker", "n_w_baker", "n_w_non_baker"]], name="baker"))
plot_data.append(get_rel_importance(weights[["w_pos", "n_w_pos", "n_w_non_pos"]], name="pos"))
plot_data.append(get_rel_importance(weights[["w_q5", "n_w_q5", "n_w_non_q5"]], name="q5"))
plot_data.append(get_rel_importance(weights[["w_gdp", "n_w_gdp", "n_w_non_gdp"]], name="gdp"))
plot_data.append(get_rel_importance(weights[["w_qsmall", "n_w_qsmall", "n_w_non_qsmall"]], name="qsmall"))
plot_data.append(get_rel_importance(weights[["w_qlarge", "n_w_qlarge", "n_w_non_qlarge"]], name="qlarge"))
plot_data.append(get_rel_importance(weights[["w_12_7"]], name="12_7"))
plot_data.append(get_rel_importance(weights[["w_6_1"]], name="6_1"))
plot_data = pd.concat(plot_data)
plot_data["time"] = "ALL"

# PRE 2003
append_data = []
append_data.append(get_rel_importance(weights[:"2002"][["w_ea", "n_w_ea", "n_w_non_ea"]], name="ea"))
append_data.append(get_rel_importance(weights[:"2002"][["w_fomc", "n_w_fomc", "n_w_non_fomc"]], name="fomc"))
append_data.append(get_rel_importance(weights[:"2002"][["w_cpi", "n_w_cpi", "n_w_non_cpi"]], name="cpi"))
append_data.append(get_rel_importance(weights[:"2002"][["w_baker", "n_w_baker", "n_w_non_baker"]], name="baker"))
append_data.append(get_rel_importance(weights[:"2002"][["w_pos", "n_w_pos", "n_w_non_pos"]], name="pos"))
append_data.append(get_rel_importance(weights[:"2002"][["w_q5", "n_w_q5", "n_w_non_q5"]], name="q5"))
append_data.append(get_rel_importance(weights[:"2002"][["w_gdp", "n_w_gdp", "n_w_non_gdp"]], name="gdp"))
append_data.append(get_rel_importance(weights[:"2002"][["w_qsmall", "n_w_qsmall", "n_w_non_qsmall"]], name="qsmall"))
append_data.append(get_rel_importance(weights[:"2002"][["w_qlarge", "n_w_qlarge", "n_w_non_qlarge"]], name="qlarge"))
append_data.append(get_rel_importance(weights[:"2002"][["w_12_7"]], name="12_7"))
append_data.append(get_rel_importance(weights[:"2002"][["w_6_1"]], name="6_1"))
append_data = pd.concat(append_data)
append_data["time"] = "Pre 2003"
plot_data = pd.concat([plot_data, append_data])

# POST 2003
append_data = []
append_data.append(get_rel_importance(weights["2003":][["w_ea", "n_w_ea", "n_w_non_ea"]], name="ea"))
append_data.append(get_rel_importance(weights["2003":][["w_fomc", "n_w_fomc", "n_w_non_fomc"]], name="fomc"))
append_data.append(get_rel_importance(weights["2003":][["w_cpi", "n_w_cpi", "n_w_non_cpi"]], name="cpi"))
append_data.append(get_rel_importance(weights["2003":][["w_baker", "n_w_baker", "n_w_non_baker"]], name="baker"))
append_data.append(get_rel_importance(weights["2003":][["w_pos", "n_w_pos", "n_w_non_pos"]], name="pos"))
append_data.append(get_rel_importance(weights["2003":][["w_q5", "n_w_q5", "n_w_non_q5"]], name="q5"))
append_data.append(get_rel_importance(weights["2003":][["w_gdp", "n_w_gdp", "n_w_non_gdp"]], name="gdp"))
append_data.append(get_rel_importance(weights["2003":][["w_qsmall", "n_w_qsmall", "n_w_non_qsmall"]], name="qsmall"))
append_data.append(get_rel_importance(weights["2003":][["w_qlarge", "n_w_qlarge", "n_w_non_qlarge"]], name="qlarge"))
append_data.append(get_rel_importance(weights["2003":][["w_12_7"]], name="12_7"))
append_data.append(get_rel_importance(weights["2003":][["w_6_1"]], name="6_1"))
append_data = pd.concat(append_data)
append_data["time"] = "Post 2003"
plot_data = pd.concat([plot_data, append_data])

# Plot
# ------- Barplot -----
fig, ax = plt.subplots(figsize=(width, 5 * cm))
sns.barplot(
    data=plot_data,
    x="type",
    y="w_mean",
    hue="time", 
    ax=ax, 
    zorder=4,
    palette=colors(3)[::-1])

ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax.axhline(1, ls="--", color="k", zorder=5)
ax.set_ylabel("Relative weight")
ax.set_xlabel("")
ax.legend(fancybox=False, edgecolor="white", loc="upper left", framealpha=1)

fig.tight_layout()
fig.savefig(os.path.join(saveLoc, f"rel_weight_barplot={int(INT_ENSEMBLE)}.pdf"), dpi=800)



# %%
# -------------- Table -------------------
table_data = plot_data[plot_data.time=="ALL"].drop(columns=["time"])

# ------- Create Table
row_names = {
    "EA": r"\text{EA}",
    "6_1": r"r\_6\_1",
    "12_7": r"r\_12\_7",
    "FOMC": r"\text{FOMC}",
    "CPI": r"\text{CPI}",
    "GDP": r"\text{GDP}",
    "BAKER": r"\text{BAKER}",
    "POS": r"r>0",
    "Q5": r"|r|>Q95",
    "QSMALL": r"r\leq Q025",
    "QLARGE": r"r\geq Q975",
}

col_names = {
    "w_mean": r"MEAN",
    "w_std": r"STD",
    "w_q10": r"Q10",
    "w_q25": r"Q25",
    "w_q50": r"Q50",
    "w_q75": r"Q75",
    "w_q90": r"Q90",
    "w_q99": r"Q99",
    "w_q999": r"Q999",
}

# - sort
table_data = table_data.set_index("type")
table_data = table_data.loc[row_names.keys(), col_names.keys()]

table = Tabularx("l" + "".join(["c"] * (table_data.shape[1])), booktabs=True)
table.add_hline()
table.add_row(
    [   MultiColumn(3, align="c", data=""),
        MultiColumn(7, align="c", data="Quantiles"),
    ]
)
table.add_hline(start=4, end=table_data.shape[1] + 1)
table.add_row([""] + [col_names[k] for k in col_names.keys()])
table.add_hline()

for idx, row in table_data.iterrows():
    to_add = [math(row_names[idx])]

    for col, num in (row.iteritems()):
        if isinstance(num, str):
            to_add.append(num)
        elif np.isnan(num) | (num == ""):
            to_add.append("")
        else:
            to_add.append(math(f"{num:.2f}"))
    table.add_row(to_add)

table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"rel_weight_ensemble={int(INT_ENSEMBLE)}"))

# %%
###########################################################################
#
#               Plot weight for example stock-day
#
###########################################################################
dt = "2009-05-01"
pm = 14593 # GME 89301
window = 2
# Create Plot data
w = att.loc[(dt, pm)]
trading_grid = pd.Series(trading_dates["date"].unique())

plot_data = trading_grid[trading_grid<w.name[0]].iloc[-w.shape[0]:]
plot_data = plot_data.to_frame("date")
plot_data["w"] = w.values
plot_data["permno"] = w.name[1]
plot_data = plot_data.reset_index(drop=True)
plot_data = plot_data.fillna(0)

#EA
ea["ea_indicator"] = 1
plot_data = plot_data.merge(
    ea.set_index(["anndats_act", "permno"])["ea_indicator"],
    left_on=["date", "permno"],
    right_index=True,
    how="left")
#Q5
r = trading_dates[trading_dates.permno == pm]
r = r[r["date"].isin(plot_data["date"])].set_index("date")
q5 = r["ret"].abs().quantile(0.95)
r["q5_indicator"] = (r["ret"].abs()>q5).astype(int)
plot_data = plot_data.merge(
    r["q5_indicator"],
    on="date",
    how="left")


# GDP
gdp["gdp_indicator"] = 1
plot_data = plot_data.merge(
    gdp.set_index("date")["gdp_indicator"],
    on="date",
    how="left")

# CPI
cpi["cpi_indicator"] = 1
plot_data = plot_data.merge(
    cpi.set_index("date")["cpi_indicator"],
    on="date",
    how="left")

# FOMC
fomc["fomc_indicator"] = 1
plot_data = plot_data.merge(
    fomc.set_index("date")["fomc_indicator"],
    on="date",
    how="left")
   

# --- BAKER mkt.abs()>2.5%
mkt_daily["baker_indicator"] = (mkt_daily["mkt"].abs()>0.025).astype(int)
plot_data = plot_data.merge(
    mkt_daily["baker_indicator"],
    on="date",
    how="left")

#plot_data = plot_data.fillna(0)
plot_data = plot_data.set_index("date")


# --- PLOT
start = plot_data.index.min()
end = plot_data.index.max() + pd.DateOffset(days=1)
fig, ax = plt.subplots(figsize=(width, 5 * cm))

for ind in ["ea_indicator", "q5_indicator", "fomc_indicator", "baker_indicator"]:#["ea_indicator", "fomc_indicator", "baker_indicator", "gdp_indicator", "cpi_indicator", "q5_indicator"]:
    tmp = plot_data[plot_data[ind]==1]
    ax.plot(
        tmp.index,
        tmp["w"].values,
        #color = "r",
        label=ind.split("_")[0],
        ls="", 
        marker="o"
    )

ax.plot(
    plot_data.index,
    plot_data["w"].values,
    color = colors(3)[2],
    label="",
    ls="-",
)

ax.set_xlabel("Date")
ax.set_ylabel("Weight")
ax.legend(fancybox=False, edgecolor="white", framealpha=1,)
ax.grid(axis="y", lw=1, ls=":", color="gray", zorder=0)
ax.set_xticks(pd.date_range(start, end, freq="2MS"))
ax.set_xlim([start, end])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.tight_layout()

# %%
###########################################################################
#
#               Spanning Regression
#
###########################################################################
# NOTE Be careful. JKP at t uses return from t+1. 
# Take this into account when mergin data 

cmp = wml_cmp["ret_cmp"].to_frame(name="cmp")
cmp = cmp.shift().dropna()       # See above

wml = wml_benchmark["ret_wml_benchmark"].to_frame(name="wml")
wml = wml.shift().dropna()       # See above

fit_data = cmp.merge(ff, on="date", how="left")
fit_data = fit_data.merge(wml, on="date", how="left")
fit_data["const"] = 1

# ------- OUR AS ENDOG
endog = "cmp"
exog = ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
reg = sm.OLS(
    endog=fit_data[endog],
    exog=fit_data[exog]
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


# --- Create table
out = reg.params.to_frame(name="CMM").T
out.loc["tvalue",:]  = reg.tvalues
out.loc["CMM","r2"] = reg.rsquared

names = {
    "const": r"\alpha",
    "Mkt-RF": r"\text{MKT}",
    "SMB": r"\text{SMB}",
    "HML": r"\text{HML}",
    "RMW": r"\text{RMW}",
    "CMA": r"\text{CMA}", 
    "Mom": r"\text{MOM}",
    "r2": r"R^2",
}

table = Tabularx("l" + "".join(["c"] * (8)), booktabs=True)
table.add_hline()
table.add_row([ "" ] + [math(names[k]) for k in out.columns])
table.add_hline()

for idx, row in out.iterrows():
    
    if idx!="tvalue":
        to_add = [idx]
        for col, num in (row.iteritems()):
            if isinstance(num, str):
                to_add.append(num)
            elif np.isnan(num) | (num == ""):
                to_add.append("")
            else:
                to_add.append(math(f"{num:.3f}"))
    else:
        to_add = [""]
        for col, num in (row.iteritems()):
            if isinstance(num, str):
                to_add.append(num)
            elif np.isnan(num) | (num == ""):
                to_add.append("")
            else:
                to_add.append(math(f"({num:.2f})"))
    
    table.add_row(to_add)

table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"spanning_reg={int(INT_ENSEMBLE)}"))



# %% 
###########################################################################
#
#               Barillas Shanken 2017
#
###########################################################################
# NOTE Be careful. JKP at t uses return from t+1. 
# Take this into account when mergin data 
# Get Data
cmp = wml_cmp["ret_cmp"].to_frame(name="cmp")
cmp = cmp.shift().dropna()       # See above

fit_data = cmp.merge(ff, on="date", how="left")

results = {}
all_fac = fit_data.columns.tolist()

for nfac in np.arange(1,len(all_fac)+1):
    best = {
        "best_sr": -100,
        "factors": ""
    }
    for facs in itertools.combinations(all_fac,nfac):  
        # Test
        fc = fit_data[list(facs)]

        cov = fc.cov()
        mean = fc.mean()
        w = np.linalg.solve(cov, mean); w /= w.sum()
        pf = (fc * w).sum(axis=1)

        sr = pf.mean()  / pf.std() * np.sqrt(12)

        if sr>best["best_sr"]:
            best["best_sr"] = sr
            best["factors"] = list(facs)
    results[nfac] = best

# Create df with x'ses
df = pd.DataFrame("",
    columns=np.arange(1,len(all_fac)+1),
    index=all_fac
)
for nfac in np.arange(1,len(all_fac)+1):
    df.loc[results[nfac]["factors"],nfac] = "x"

# ORdering
# ordering = (df=="x").sum(axis=1).sort_values(ascending=False).index
# df = df.loc[ordering,:]
# %%
# Create table
names = {
    "cmp": r"\text{CMM}",
    "Mkt-RF": r"\text{MKT}",
    "SMB": r"\text{SMB}",
    "HML": r"\text{HML}",
    "RMW": r"\text{RMW}",
    "CMA": r"\text{CMA}", 
    "Mom": r"\text{MOM}",
}
n_facs_to_show = list(np.arange(2,len(all_fac)+1))
df = df.loc[:,n_facs_to_show]
results = {k: results[k] for k in n_facs_to_show}

table = Tabularx("l" + "".join(["c"] * (len(n_facs_to_show))), booktabs=True)
table.add_hline()
table.add_row([ "" , MultiColumn(len(n_facs_to_show), align="c", data="Number of Factors"),])
table.add_hline(start=2, end=len(n_facs_to_show)+1)
table.add_row([ "" ] + [str(int(f)) for f in n_facs_to_show])
table.add_hline()

# Add SR
to_add = ["SR"]
for idx, row in results.items():
    num = row["best_sr"]
    if isinstance(num, str):
        to_add.append(num)
    elif np.isnan(num) | (num == ""):
        to_add.append("")
    else:
        to_add.append(math(f"{num:.3f}"))
table.add_row(to_add)

# ADD INCLUDED FACS
table.add_hline()
table.add_row(["", MultiColumn(len(n_facs_to_show), align="c", data="Included Factors")])
table.add_hline()

for idx,row in df.iterrows():
    to_add = [math(names[idx])]
    for col, num in row.items():
            to_add.append(num)
    table.add_row(to_add)
table.add_hline()
# create .tex
table.generate_tex(os.path.join(saveLoc, f"BS17={int(INT_ENSEMBLE)}"))

















































# %%
###########################################################################
#
#              Fama-MacBeth
#
###########################################################################
jkp = pd.read_parquet("../03_data/kelly_characteristics/jkp.pq", columns=["ret_exc_lead1m"])

# Get all factors
cmp = wml_cmp["ret_cmp"].to_frame(name="cmp")
cmp = cmp.shift().dropna()       # See above

wml = wml_benchmark["ret_wml_benchmark"].to_frame(name="wml")
wml = wml.shift().dropna()       # See above

factors = cmp.merge(ff, on="date", how="left")
factors = factors.merge(wml, on="date", how="left")
factors["const"] = 1

# crop to same time span
min_date = factors.index.min() - pd.DateOffset(months=1)
max_date = "2015"
jkp = jkp[min_date:]
jkp = jkp[:max_date]

# Calcualte betas
def calc_betas(df, factors):
    exog = ["const",  "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    endog = ["ret_exc_lead1m"]
    pm = df.index.get_level_values("permno")[0]

    # Merge factors
    # NOTE Same argument applies as above. --> shift kelly data
    df = df.shift()
    df = df.merge(factors, on="date", how="right")
    df = df.dropna(subset=["ret_exc_lead1m"])

    # Require at least 60 obs.
    if df.shape[0]>=60:
        # Calculate betas
        reg = sm.OLS(
            exog=df[exog],
            endog=df[endog],
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

        beta = reg.params.to_frame(name=pm).T
        beta.index.name = "permno"

        tvalues = reg.tvalues.to_frame(name=pm).T
        tvalues.index.name = "permno"
    else:
        beta = pd.DataFrame(columns=exog, index=[pm])
        beta.index.name = "permno"

        tvalues = pd.DataFrame(columns=exog, index=[pm])
        tvalues.index.name = "permno"
    
    return beta, tvalues

# Calculate lambdas
def calc_lambdas(df,betas):
    dte = df.index.get_level_values("date")[0] + pd.DateOffset(months=1)

    # Merge
    df = df.merge(betas, on="permno", how="right")
    df = df.dropna()

    # Regression
    exog = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    endog = ["ret_exc_lead1m"]

    reg = sm.OLS(
        exog=df[exog],
        endog=df[endog],
        ).fit()
    
    lambdas = reg.params.to_frame(name=dte).T
    lambdas.index.name = "date"

    tvalues = reg.tvalues.to_frame(name=dte).T
    tvalues.index.name = "date"

    return lambdas, tvalues

# ------ BETAS
fn = os.path.join(saveLoc, f"fmb_betas={int(INT_ENSEMBLE)}")
if os.path.exists(fn):
    betas = pd.read_parquet(fn)
else:
    res = Parallel(n_jobs=CORES,verbose=True)(
        delayed(calc_betas)(jkp[jkp.index.get_level_values("permno")==pm],factors) for pm in jkp.index.get_level_values("permno").unique()
    )
    betas = pd.concat([d[0]  for d in res])
    betas = betas.dropna(how="all", axis=0)
    betas.to_parquet(fn)
    tvalues = pd.concat([d[1]  for d in res])
    tvalues = tvalues.dropna(how="all", axis=0)
    tvalues.to_parquet(os.path.join(saveLoc, f"fmb_betas_tvalues={int(INT_ENSEMBLE)}"))

# ----- Lambdas
fn = os.path.join(saveLoc, f"fmb_lambdas={int(INT_ENSEMBLE)}")
if os.path.exists(fn):
    lambdas = pd.read_parquet(fn)
else:
    res = Parallel(n_jobs=CORES,verbose=True)(
        delayed(calc_lambdas)(jkp[jkp.index.get_level_values("date")==dte],betas) for dte in jkp.index.get_level_values("date").unique()
    )
    lambdas = pd.concat([d[0]  for d in res])
    lambdas = lambdas.dropna(how="all", axis=0)
    lambdas.to_parquet(fn)
    tvalues = pd.concat([d[1]  for d in res])
    tvalues = tvalues.dropna(how="all", axis=0)
    tvalues.to_parquet(os.path.join(saveLoc, f"fmb_lambdas_tvalues={int(INT_ENSEMBLE)}"))

# Analysis
final = pd.DataFrame()
for col in lambdas.columns:
    reg = sm.OLS(
        exog = np.ones(lambdas.shape[0]),
        endog = lambdas[col]
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    final.loc[col, "mean"] = reg.params["const"]*12
    final.loc[col, "tvalue"] = reg.tvalues["const"]



# %%
###########################################################################
#
#              Fama-MacBeth (wit Portfolios) 
#
###########################################################################

# FF portfolios
#BE_BEME
p_s_v = read_ff("25_Portfolios_5x5")
p_s_v.index = p_s_v.index.to_timestamp()
p_s_v =  p_s_v.stack().to_frame("ret")
p_s_v.index.names = ("date", "id")
# 48 Industries
p_ind = read_ff("48_Industry_Portfolios")
p_ind.index = p_ind.index.to_timestamp()
p_ind =  p_ind.stack().to_frame("ret")
p_ind.index.names = ("date", "id")
# BE_MOM
p_s_mom = read_ff("25_Portfolios_ME_Prior_12_2") 
p_s_mom.index = p_s_mom.index.to_timestamp()
p_s_mom =  p_s_mom.stack().to_frame("ret")
p_s_mom.index.names = ("date", "id")

portfolios = pd.concat([p_s_v, p_ind, p_s_mom])

# Get all factors
cmp = wml_cmp["ret_cmp"].to_frame(name="cmp")
cmp = cmp.shift().dropna()       # See above

wml = wml_benchmark["ret_wml_benchmark"].to_frame(name="wml")
wml = wml.shift().dropna()       # See above

factors = cmp.merge(ff, on="date", how="left")
factors = factors.merge(wml, on="date", how="left")
factors["const"] = 1

# crop to same time span
portfolios = portfolios.sort_index()
min_date = factors.index.min() - pd.DateOffset(months=1)
max_date = "2023"
min_date = "2003"
portfolios = portfolios[min_date:]
portfolios = portfolios[:max_date]

# Calcualte betas
def calc_betas(df, factors):
    exog = ["const", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "cmp"]
    endog = ["ret"]
    pm = df.index.get_level_values("id")[0]

    # Merge factors
    df = df.merge(factors, on="date", how="right")
    df = df.dropna(subset=["ret"])

    # Require at least 60 obs.
    if df.shape[0]>=60:
        # Calculate betas
        reg = sm.OLS(
            exog=df[exog],
            endog=df[endog],
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

        beta = reg.params.to_frame(name=pm).T
        beta.index.name = "id"

        tvalues = reg.tvalues.to_frame(name=pm).T
        tvalues.index.name = "id"
    else:
        beta = pd.DataFrame(columns=exog, index=[pm])
        beta.index.name = "id"

        tvalues = pd.DataFrame(columns=exog, index=[pm])
        tvalues.index.name = "id"
    
    return beta, tvalues

# ------ BETAS
res = Parallel(n_jobs=CORES,verbose=True)(
    delayed(calc_betas)(portfolios[portfolios.index.get_level_values("id")==pm],factors) for pm in portfolios.index.get_level_values("id").unique()
)
betas = pd.concat([d[0]  for d in res])
betas = betas.dropna(how="all", axis=0)
tvalues = pd.concat([d[1]  for d in res])
tvalues = tvalues.dropna(how="all", axis=0)


# Calculate lambdas
def calc_lambdas(df,betas):
    dte = df.index.get_level_values("date")[0] 

    # Merge
    df = df.merge(betas, on="id", how="right")
    df = df.dropna()

    # Regression
    #df["const"] = 1
    exog = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    endog = ["ret"]

    reg = sm.OLS(
        exog=df[exog],
        endog=df[endog],
        ).fit()
    
    lambdas = reg.params.to_frame(name=dte).T
    lambdas.index.name = "date"

    tvalues = reg.tvalues.to_frame(name=dte).T
    tvalues.index.name = "date"

    return lambdas, tvalues


# ----- Lambdas
res = Parallel(n_jobs=CORES,verbose=True)(
    delayed(calc_lambdas)(portfolios[portfolios.index.get_level_values("date")==dte],betas) for dte in portfolios.index.get_level_values("date").unique()
)
lambdas = pd.concat([d[0]  for d in res])
lambdas = lambdas.dropna(how="all", axis=0)

tvalues = pd.concat([d[1]  for d in res])
tvalues = tvalues.dropna(how="all", axis=0)


# Analysis
final = pd.DataFrame()
for col in lambdas.columns:
    reg = sm.OLS(
        exog = np.ones(lambdas.shape[0]),
        endog = lambdas[col]
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    final.loc[col, "mean"] = reg.params["const"]*12
    final.loc[col, "tvalue"] = reg.tvalues["const"]
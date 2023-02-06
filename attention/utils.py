# %%
import time
from joblib import dump, load
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

from ray import tune

# %%
def timed_callout(callout: str):
    print(time.ctime() + " ::: " + str(callout), flush=True)


# %%
def save_config(config, logLoc: str, add_params: dict = {}):
    # ---- save config (.pkl) and overview (.json) of this trial
    for key, value in add_params.items():
        config["hyper_params"][key] = value
    with open(os.path.join(logLoc, "overview.json"), "w") as fp:
        json.dump(config["hyper_params"], fp)
    dump(config["hyper_params"], os.path.join(logLoc, "config.pkl"))


# %%
def trial_name_creator(trial):
    """
        Required on Palma II because of restrictions in the maximum length of a given file's name.
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    return str(trial) + "_" + str(trial.experiment_tag[:25])


#%%
def get_checkpoints_paths(logdir):
    """Finds the checkpoints within a specific folder.
    Returns a pandas DataFrame of training iterations and checkpoint
    paths within a specific folder.
    Raises:
        FileNotFoundError if the directory is not found.
    """
    marker_paths = glob.glob(os.path.join(logdir, "checkpoint_*/.is_checkpoint"))
    iter_chkpt_pairs = []
    for marker_path in marker_paths:
        chkpt_dir = os.path.dirname(marker_path)
        metadata_file = os.path.join(chkpt_dir, ".tune_metadata")
        if not os.path.isfile(metadata_file):
            print("{} has zero or more than one tune_metadata.".format(chkpt_dir))
            # return None
        tmp_marker = os.path.join(chkpt_dir, ".temp_marker")
        if os.path.isfile(tmp_marker):
            print("{} marked as temporary.".format(chkpt_dir))
            # continue
            # return None
        chkpt_path = metadata_file[: -len(".tune_metadata")]
        chkpt_iter = int(chkpt_dir[chkpt_dir.rfind("_") + 1 :])
        iter_chkpt_pairs.append([chkpt_iter, chkpt_path])
    chkpt_df = pd.DataFrame(iter_chkpt_pairs, columns=["training_iteration", "chkpt_path"])
    return chkpt_df

# %%
def load_nbest_trials(
    logLoc: str = "", n_best_models: int = 10, grace_period: int = None, grid_variables: list = []
):
    if logLoc == "":
        logLocs = glob.glob("../05_models/*")
        logLocs = [loc for loc in logLocs if not loc.endswith(".xlsx")]
        for i, loc in enumerate(logLocs):
            print("%d: %s" % (i, loc))
        logLoc = logLocs[int(input("Which log location? [Number]: "))]

    print(f"Retrieving trials for TuneExperiment at {logLoc}.")

    model_config = load(os.path.join(logLoc, "config.pkl"))
    trialLoc = os.path.join(logLoc, os.path.split(model_config["trialLoc"])[-1])
    # trialLoc = model_config["hyper_params"]["trialLoc"]

    # save best config dataframe:
    analysis = tune.ExperimentAnalysis(trialLoc)
    dfs = analysis.trial_dataframes
    cfs = analysis.get_all_configs()
    best_df = []
    for trial in dfs.keys():
        if len(dfs[trial]) > 0:
            tmp = dfs[trial]
            if "score" not in tmp.columns:  # faulty recording, skip.
                continue
            tmp["path"] = trial
            cf = pd.Series(cfs[trial]).to_frame().T
            cf.columns = ["config/" + str(c) for c in cf.columns]
            tmp = pd.concat((tmp, cf), axis=1)
            for col in tmp.columns:
                if "config/" in col:
                    tmp[col] = tmp[col].ffill()
            best_df.append(tmp)
    best_df = pd.concat(best_df).reset_index(drop=True)

    if grace_period is None:
        best_df = best_df[
            best_df["training_iteration"] > best_df["config/tune_params"].apply(pd.Series)["trial_grace_period"]
        ]
    else:
        best_df = best_df[best_df["training_iteration"] > grace_period]
    best_df = best_df.sort_values("score")

    # ---- retrieve up to "n_models" best checkpoints save in logLoc.
    best_models = best_df.groupby("trial_id").first().sort_values("score")

    # obtain best trial per val_start_month in case we use num_samples > 1
    hyper_params = best_models["config/hyper_params"].apply(pd.Series)
    best_models = best_models.merge(hyper_params, on="trial_id")
    if grid_variables:
        best_models["rank"] = best_models.groupby(grid_variables).score.rank(ascending=True)
    else:
        best_models["rank"] = best_models.score.rank(ascending=True)
    # best_models = best_models.reset_index().groupby("val_start_month", as_index=False).first().set_index("trial_id")

    best_models = best_models[best_models["rank"] <= n_best_models]

    return best_models


# %%
def save_best_trials(
    logLoc: str = "", n_best_models_to_save: int = 10, grace_period: int = None, grid_variables: list = []
):
    if logLoc == "":
        logLocs = glob.glob("../05_models/*")
        logLocs = [loc for loc in logLocs if not loc.endswith(".xlsx")]
        for i, loc in enumerate(logLocs):
            print("%d: %s" % (i, loc))
        logLoc = logLocs[int(input("Which log location? [Number]: "))]

    print(f"Retrieving trials for TuneExperiment at {logLoc}.")

    model_config = load(os.path.join(logLoc, "config.pkl"))
    trialLoc = os.path.join(logLoc, os.path.split(model_config["trialLoc"])[-1])
    # trialLoc = model_config["hyper_params"]["trialLoc"]

    # save best config dataframe:
    analysis = tune.ExperimentAnalysis(trialLoc)
    dfs = analysis.trial_dataframes
    cfs = analysis.get_all_configs()
    best_df = []
    for trial in dfs.keys():
        if len(dfs[trial]) > 0:
            tmp = dfs[trial]
            if "score" not in tmp.columns:  # faulty recording, skip.
                continue
            tmp["path"] = trial
            cf = pd.Series(cfs[trial]).to_frame().T
            cf.columns = ["config/" + str(c) for c in cf.columns]
            tmp = pd.concat((tmp, cf), axis=1)
            for col in tmp.columns:
                if "config/" in col:
                    tmp[col] = tmp[col].ffill()
            best_df.append(tmp)
    best_df = pd.concat(best_df).reset_index(drop=True)

    if grace_period is None:
        best_df = best_df[
            best_df["training_iteration"] > best_df["config/tune_params"].apply(pd.Series)["trial_grace_period"]
        ]
    else:
        best_df = best_df[best_df["training_iteration"] > grace_period]
    best_df = best_df.sort_values("score")
    best_df.to_pickle(os.path.join(logLoc, "all_trials.pkl"))

    # ---- retrieve up to "n_models" best checkpoints save in logLoc.
    best_models = best_df.groupby("trial_id").first().sort_values("score")

    # obtain best trial per val_start_month in case we use num_samples > 1
    hyper_params = best_models["config/hyper_params"].apply(pd.Series)
    best_models = best_models.merge(hyper_params, on="trial_id")
    if grid_variables:
        best_models["rank"] = best_models.groupby(grid_variables).score.rank(ascending=True)
    else:
        best_models["rank"] = best_models.score.rank(ascending=True)
    # best_models = best_models.reset_index().groupby("val_start_month", as_index=False).first().set_index("trial_id")

    best_models = best_models[best_models["rank"] <= n_best_models_to_save]

    print(best_models[["score", "training_iteration"]].head(20), flush=True)
    for i, (idx, row) in enumerate(best_models.iterrows()):  # copy n models over
        print("Copying best trials (%d-best model, path=%s)" % (i, idx))
        if i == 0:
            for key in row.index:  # config keys
                if key.startswith("config"):
                    print(f" --- {key} --- ")
                    if isinstance(row[key], dict):
                        for inner in row[key]:
                            print(f"\t{inner}: {row[key][inner]}")
                    else:
                        print(pd.Series(row[key]))
                    print("\n")

        best_iteration = row["training_iteration"] - 1
        paths = get_checkpoints_paths(row["path"])
        print("Looking for best iteration %d." % best_iteration)
        if isinstance(paths, pd.DataFrame):
            path = paths.loc[(paths["training_iteration"] - best_iteration).abs().idxmin(), "chkpt_path"]
            # val_start_month = row["config/hyper_params"]["val_start_month"]
            # file = row["config/hyper_params"]["file"].replace(".pq", "")
            rank = row["rank"]
            # copyfile(path + "checkpoint", os.path.join(logLoc, "best_%d" % n_models_transfered))
            specifier = "best_" + "_".join(row["config/hyper_params"][var] for var in grid_variables)
            print(f"{specifier}")
            # copyfile(path + "checkpoint", os.path.join(logLoc, f"{val_start_month}_{file}_{rank}"))
            copyfile(path + "checkpoint", os.path.join(logLoc, f"{specifier}_{int(rank)}"))

            # create error plot
            scores = dfs[row.path].sort_values("training_iteration")
            n_lookback = 50
            best_score = scores["score"].min()
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(scores["train_score"], ls="-", color="k")
            ax[0].plot(scores["score"], ls="-", color="r")
            ax[0].axhline(np.min(scores["train_score"]), ls="--", color="k", lw=1)
            ax[0].axhline(np.min(scores["score"]), ls="--", color="r", lw=1)
            ax[0].legend(["Train", "Val"])
            ax[0].set_title(f"Best Validation Score: {best_score:.4f}")

            ax[1].plot(scores["train_score"].iloc[-n_lookback:], ls=":", marker="o", color="k")
            ax[1].plot(scores["score"].iloc[-n_lookback:], ls=":", marker="o", color="r")
            ax[1].axhline(np.min(scores["train_score"]), ls="--", color="k", lw=1)
            ax[1].axhline(np.min(scores["score"]), ls="--", color="r", lw=1)
            ax[1].legend(["Train", "Val"])

            plt.tight_layout()
            fig.savefig(os.path.join(logLoc, f"{specifier}_{int(rank)}.pdf"), dpi=800)

            # Create R2 Plot
            if "r2" in scores.columns:
                best_r2 = scores["r2"].max()
                fig, ax = plt.subplots(2, 1, figsize=(8, 6))
                ax[0].plot(scores["train_r2"], ls="-", color="k")
                ax[0].plot(scores["r2"], ls="-", color="r")
                ax[0].axhline(np.max(scores["train_r2"]), ls="--", color="k", lw=1)
                ax[0].axhline(0, ls="-", color="k", lw=1)
                ax[0].axhline(np.max(scores["r2"]), ls="--", color="r", lw=1)
                ax[0].legend(["Train", "Val"])
                ax[0].set_title(f"Best Validation R2: {best_r2:.4f}")

                ax[1].plot(scores["train_r2"].iloc[-n_lookback:], ls=":", marker="o", color="k")
                ax[1].plot(scores["r2"].iloc[-n_lookback:], ls=":", marker="o", color="r")
                ax[1].axhline(np.max(scores["train_r2"]), ls="--", color="k", lw=1)
                ax[1].axhline(0, ls="-", color="k", lw=1)
                ax[1].axhline(np.max(scores["r2"]), ls="--", color="r", lw=1)
                ax[1].legend(["Train", "Val"])
                plt.tight_layout()
                fig.savefig(os.path.join(logLoc, f"{specifier}_R2_{int(rank)}.pdf"), dpi=800)

            if "epd" in scores.columns:
                best_epd = scores["epd"].min()
                fig, ax = plt.subplots(2, 1, figsize=(8, 6))
                ax[0].plot(scores["train_epd"], ls="-", color="k")
                ax[0].plot(scores["epd"], ls="-", color="r")
                ax[0].axhline(np.min(scores["train_epd"]), ls="--", color="k", lw=1)
                ax[0].axhline(np.min(scores["epd"]), ls="--", color="r", lw=1)
                ax[0].legend(["Train", "Val"])
                ax[0].set_title(f"Best Validation EPD: {best_epd:.2f}")

                ax[1].plot(scores["train_epd"].iloc[-n_lookback:], ls=":", marker="o", color="k")
                ax[1].plot(scores["epd"].iloc[-n_lookback:], ls=":", marker="o", color="r")
                ax[1].axhline(np.min(scores["train_epd"]), ls="--", color="k", lw=1)
                ax[1].axhline(np.min(scores["epd"]), ls="--", color="r", lw=1)
                ax[1].legend(["Train", "Val"])
                plt.tight_layout()
                fig.savefig(os.path.join(logLoc, f"{specifier}_EPD_{int(rank)}.pdf"), dpi=800)

        else:
            best_models.loc[idx] = np.nan  # gets rid of faulty trials + temporary trials.

    best_models.dropna(how="all").to_pickle(os.path.join(logLoc, "configs.pkl"))

# %%
from socket import gethostname
import os
import glob
import re
from functools import partial


import datetime
import time
import pandas as pd
import numpy as np
from pathlib import Path
import cloudpickle

# -- TORCH
import torch
import torch.nn as nn

# -- RAY
import ray
from ray.train.torch import TorchConfig
import ray.train as train
from ray.train import Trainer



# -- MODEL
from attention.data_loading import ModelDataset_CharModel_OOS, MyDistributedSampler

from attention.char_model import CharacteristicWeightedMomentum
from attention.utils import timed_callout, load_nbest_trials, save_best_trials
from attention.loop_CharModel import test_epoch



# %%
def test_func(config):
    #print(config)

    # num_workers = config["hyper_params"]["num_workers"]
    val_pct = config["hyper_params"]["val_pct"]
    batch_pct = config["hyper_params"]["batch_pct"]
    start_test_month = config["hyper_params"]["start_test_month"]

    # ---- DATASETS
    test_data = ModelDataset_CharModel_OOS(
        filename="../04_results/Momentum_DataSet_TargetOnly=False_std=ranks.pq",
        val_pct=val_pct,
        name="test",
        start_test_month=start_test_month,
    )

    timed_callout(f"Test sample size: {len(test_data)}")

    # SET BATCH SIZE (Multiple of 8)
    batch_size = (round(len(test_data) * batch_pct) // 8) * 8 + 8
    timed_callout(f"Batch size: {batch_size}.")

    # SAMPLER
    sampler = MyDistributedSampler(dataset=test_data)

    # LOADER
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=0, pin_memory=True, sampler=sampler
    )

    # ALL INPUT MUST BE ON THE SAME DEVICE
    loader = train.torch.prepare_data_loader(loader, add_dist_sampler=False)


    idx_best = int(config["rank"])
    checkpoint_to_load = os.path.join(logLoc, f"best__{idx_best}")
    with Path(checkpoint_to_load).expanduser().open("rb") as f:
        model_state = cloudpickle.load(f)
   

    # MODEL
    timed_callout("Setting up model.")
    model = CharacteristicWeightedMomentum(**config["model_params"])


    # NOTE Clarify with Heiner why he does this.
    for key in list(model_state["model_state"].keys()):
        if key[:7] == "module.":
            model_state["model_state"][key[7:]] = model_state["model_state"].pop(key)
    model.load_state_dict(model_state["model_state"])
    model.eval()  # evaluation mode, fixed batch norm and dropout layers.

    model = train.torch.prepare_model(model)

    # Run testing
    pred, att = test_epoch(dataLoader=loader, model=model)

    # Saving
    # saveLoc = os.path.join(logLoc, "results")
    # os.makedirs(saveLoc, exist_ok=True)
    # pred.sort_values("date").to_parquet(os.path.join(saveLoc, "predicted_values.pq"))
    # att.sort_values("date").to_parquet(os.path.join(saveLoc, "attention_weights.pq"))

    train.report(score=1)

    return {"pred": pred.sort_values("date"), "att": att.sort_values("date")}


# %%
if __name__ == "__main__":
    LOCAL = True
    BACKEND = "gloo"
    NUM_WORKERS = 1
    N_ENSEMBLE = 3
    #MODEL = "_OOS_models_CharWeighted_20230113_1408"
    MODEL = "_OOS_models_CharWeighted_20230120_1604"

    save_loc = os.path.join("../05_models", MODEL)

    # Get all relevant folders
    fns = glob.glob(os.path.join(save_loc, "Char*"))

    for logLoc in fns:
        # Get start_test_month
        start_test_month = pd.to_datetime(re.search("\d{4}-\d{2}-\d{2}", logLoc).group(), format="%Y-%m-%d")
        

        # START RAY
        try:
            if LOCAL:
                ray.init(include_dashboard=False)
            else:
                os.makedirs(os.environ["RAY_TMPDIR"], exist_ok=True)
                ray.init(
                    _temp_dir=os.environ["RAY_TMPDIR"],
                    adress=os.environ["ip_head"],
                    _redis_max_memory=1024**1,
                )
                timed_callout("Ray Cluster started.")
        except Exception as e:
            print("Ray already started.", e)


        # SAVE N best trials (if not done before)
        save_best_trials(logLoc=logLoc, n_best_models_to_save=N_ENSEMBLE)

        # ---- RUN Testing
        start_time = time.time()

        best_df = load_nbest_trials(logLoc=logLoc, n_best_models=N_ENSEMBLE)
        N = best_df.shape[0]

        config_list = best_df.loc[:, [col for col in best_df.columns if ("config/" in col) | ("rank" in col)]]
        config_list.columns = [c.replace("config/", "") for c in config_list.columns]
        config_list = [x._asdict() for x in config_list.itertuples()]

        trainer = Trainer(
            backend=TorchConfig(backend=BACKEND),
            num_workers=NUM_WORKERS,
            use_gpu=True if torch.cuda.device_count() > 0 else False,
            logdir=logLoc,
        )
        trainer.start()
        prediction = pd.DataFrame()
        attention = pd.DataFrame()
        for config in config_list:
            res = trainer.run(test_func, config,   checkpoint_strategy=None, callbacks=None, checkpoint=None)
            prediction = pd.concat([prediction, res[0]["pred"]])
            attention = pd.concat([attention, res[0]["att"]])
        trainer.shutdown()

        # Calculate mean
        prediction = prediction.groupby(["date", "id"]).mean().reset_index()
        attention = attention.groupby(["date", "id"]).mean().reset_index()

        # Saving
        saveLoc = os.path.join(logLoc, "results")
        os.makedirs(saveLoc, exist_ok=True)
        prediction.to_parquet(os.path.join(saveLoc, f"Ensemble_N={N}_predicted_values.pq"))
        attention.to_parquet(os.path.join(saveLoc, f"Ensemble_N={N}_attention_weights.pq"))


        estimation_time = (time.time() - start_time) / 60
        timed_callout(f" Testing took {estimation_time:.3f} minutes")

        # cleanup
        ray.shutdown()
        time.sleep(10)

# %%

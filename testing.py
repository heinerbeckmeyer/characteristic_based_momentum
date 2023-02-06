# %%
from socket import gethostname
import os

# os.environ['MASTER_ADDR'] = 'fe80::d8ea:4ce7:7c8a:cbcd'
# os.environ['MASTER_PORT'] = '9999'

import datetime
import time
import pandas as pd
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
from ray import tune
from ray.tune.suggest import BasicVariantGenerator


# -- MODEL
from attention.data_loading import ModelDataset, MyDistributedSampler

from attention.model import SingleHeadAttention
from attention.utils import (
    timed_callout,
)
from attention.loop import test_epoch



# %%
def test_func(config):
    print(config)

    num_workers = config["hyper_params"]["num_workers"]
    val_pct = config["hyper_params"]["val_pct"]
    batch_pct = config["hyper_params"]["batch_pct"]

    # ---- DATASETS
    test_data = ModelDataset(
        filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
        val_pct=val_pct,
        name="test",
    )


    timed_callout(f"Full sample size: {len(test_data)}")

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

    # MODEL
    timed_callout("Setting up model.")

    checkpoint_to_load = os.path.join(logLoc, f"best__1")
    with Path(checkpoint_to_load).expanduser().open("rb") as f:
        model_state = cloudpickle.load(f)
    config["model_params"]["retrieve_attention_mask"] = True

    # MODEL
    timed_callout("Setting up model.")
    if "SingleHeadAttention" in config["hyper_params"]["trialLoc"]:
        model = SingleHeadAttention(**config["model_params"])
    else:
        raise ValueError("Selected model not implemented yet.")

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
    saveLoc = os.path.join(logLoc, "results")
    os.makedirs(saveLoc, exist_ok=True)
    pred.sort_values("date").to_parquet(os.path.join(saveLoc, f"predicted_values.pq"))

    att.sort_values("date").to_parquet(os.path.join(saveLoc, f"attention_weights.pq"))



    train.report(score=1)

    return
# %%
if __name__ == "__main__":

    MODEL_LOC = "../05_models"
    TRIAL_NAME = "SingleHeadAttention_posEnc=False___20221021_1458"
    BACKEND = "gloo"  #"nccl"        # Shall we use gloo?

    HOST = gethostname()
    LOCAL = True
    NUM_WORKERS = 1

    # ---- Config
    timed_callout(f"Testing trial {TRIAL_NAME}.")
    logLoc = os.path.abspath(os.path.join(MODEL_LOC, str(TRIAL_NAME)))
    trial_name = TRIAL_NAME

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


    # ---- RUN Testing
    start_time = time.time()

    best_df = pd.read_pickle(os.path.join(logLoc, "configs.pkl"))

    # Get best score
    best_df = best_df[best_df.score == best_df.score.min()]

    config_list = best_df.loc[:, [col for col in best_df.columns if "config/" in col]]
    config_list.columns = [c.replace("config/", "") for c in config_list.columns]
    config = [x._asdict() for x in config_list.itertuples()][0]

    if "pos_encoding" in config["model_params"]:
        config["model_params"]["do_pos_encoding"] = config["model_params"].pop("pos_encoding")

    trainer = Trainer(
        backend=TorchConfig(backend=BACKEND),
        num_workers=NUM_WORKERS,
        use_gpu=True if torch.cuda.device_count() > 0 else False,
        logdir=logLoc,
    )
    trainer.start()

    trainer.run(test_func, config, checkpoint_strategy=None, callbacks=None, checkpoint=None)
    trainer.shutdown()


    estimation_time = (time.time() - start_time) / 60
    timed_callout(f" Testing took {estimation_time:.3f} minutes")
    
    # cleanup
    ray.shutdown()

# %%

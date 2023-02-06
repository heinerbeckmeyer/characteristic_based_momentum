# %%
from socket import gethostname
import os

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
    save_config,
    trial_name_creator,
    save_best_trials,
)
from attention.early_stopping import ExperimentAndTrialPlateauStopper
from attention.callbacks import MyCallback
from attention.loop import train_epoch, validate_epoch, test_epoch


# %%
def train_func(config):
    print(config)

    epochs = config["hyper_params"]["epochs"]
    num_workers = config["hyper_params"]["num_workers"]
    val_pct = config["hyper_params"]["val_pct"]
    batch_pct = config["hyper_params"]["batch_pct"]

    # ---- DATASETS
    train_data = ModelDataset(
        filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
        val_pct=val_pct,
        std_char=config["data_params"]["std_char"],
        name="train",
    )
    val_data = ModelDataset(
        filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
        val_pct=val_pct,
        std_char=config["data_params"]["std_char"],
        name="val",
    )

    timed_callout(f"Train-data size: {len(train_data)} - Val-data size: {len(val_data)}")

    # SET BATCH SIZE (Multiple of 8)
    batch_size = (round(len(train_data) * batch_pct) // 8) * 8 + 8
    timed_callout(f"Batch size: {batch_size}.")

    # SAMPLER
    train_sampler = MyDistributedSampler(dataset=train_data)
    val_sampler = MyDistributedSampler(dataset=val_data)

    # LOADER
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=0, pin_memory=True, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size * 4, num_workers=0, pin_memory=True, sampler=val_sampler
    )

    # ALL INPUT MUST BE ON THE SAME DEVICE
    train_loader = train.torch.prepare_data_loader(train_loader, add_dist_sampler=False)
    val_loader = train.torch.prepare_data_loader(val_loader, add_dist_sampler=False)

    # MODEL
    timed_callout("Setting up model.")
    if "SingleHeadAttention" in config["hyper_params"]["trialLoc"]:
        model = SingleHeadAttention(**config["model_params"])
    else:
        raise ValueError("Selected model not implemented yet.")
    model = train.torch.prepare_model(model)

    # LOSS
    loss_fn = nn.MSELoss()

    # OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), **config["optim_params"])

    # SCHEDULER
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        **config["scheduler_params"],
        epochs=epochs,
        steps_per_epoch=(len(train_data) // batch_size // num_workers) + 1,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     **config["scheduler_params"],
    # )

    # ---- RESUME?
    checkpoint = train.load_checkpoint() or {}
    if checkpoint:
        timed_callout("Restoring checkpoint")
        model.load_state_dict(checkpoint["model_sate"])
        optimizer.load_state_dict(checkpoint["optim_sate"])
        scheduler.load_state_dict(checkpoint["scheduler_sate"])
        current_epoch = checkpoint["epoch"] + 1
    else:
        current_epoch = 0

    for epoch in range(current_epoch, epochs):
        train_score, train_r2 = train_epoch(
            dataLoader=train_loader,
            model=model,
            loss_fn=loss_fn,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        score, r2 = validate_epoch(
            dataLoader=val_loader,
            model=model,
            loss_fn=loss_fn,
            epoch=epoch,
        )
        train.save_checkpoint(
            **{
                "epoch": epoch,
                "_training_iteration": epoch,
                "score": score,
                "train_score": train_score,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
        )

        train_reports = dict(
            score=score,
            train_score=train_score,
            r2=r2,
            train_r2=train_r2,
            lr=optimizer.param_groups[0]["lr"],
        )

        # train.report(score=score,train_score=train_score,lr=optimizer.param_groups[0]["lr"])
        train.report(**train_reports)

    return


# %%
def test_func(config):
    print(config)

    # num_workers = config["hyper_params"]["num_workers"]
    val_pct = config["hyper_params"]["val_pct"]
    batch_pct = config["hyper_params"]["batch_pct"]

    # ---- DATASETS
    test_data = ModelDataset(
        filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
        val_pct=val_pct,
        std_char=config["data_params"]["std_char"],
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

    checkpoint_to_load = os.path.join(logLoc, "best__1")
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
    pred.sort_values("date").to_parquet(os.path.join(saveLoc, "predicted_values.pq"))

    att.sort_values("date").to_parquet(os.path.join(saveLoc, "attention_weights.pq"))

    train.report(score=1)

    return


# %%
if __name__ == "__main__":

    MODEL_LOC = "../05_models"
    MODEL_TYPE = "SingleHeadAttention"
    POS_ENCODING = True
    BACKEND = "gloo"  
    STDCHAR = True

    HOST = gethostname()
    LOCAL = True
    NUM_WORKERS = 1
    RESUME_TRIAL = ""

    # ---- Config
    if RESUME_TRIAL != "":
        timed_callout(f"Resuming trial {RESUME_TRIAL}.")
        logLoc = os.path.abspath(os.path.join(MODEL_LOC, str(RESUME_TRIAL)))
        trial_name = RESUME_TRIAL
    else:
        timed_callout("Starting new model estimation.")
        trial_number = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        trial_name = MODEL_TYPE + f"_posEnc={POS_ENCODING}_stdChar={STDCHAR}" + "___" + str(trial_number)
        logLoc = os.path.abspath(
            os.path.join(
                MODEL_LOC,
                trial_name,
            )
        )
        os.makedirs(logLoc, exist_ok=True)
    os.makedirs(os.path.join(logLoc, "logs"), exist_ok=True)

    config = {}
    # HYPERPARAMETER
    config["hyper_params"] = dict(
        num_workers=NUM_WORKERS,
        num_samples=1,
        epochs=150,
        trialLoc=os.path.join(logLoc, trial_name),
        val_pct=0.3,
        batch_pct=0.01,
    )

    config["data_params"] = dict(
        std_char=STDCHAR
    )

    # MODEL PARAMETER
    config["model_params"] = dict(
        n_lags=231,
        alpha=1.0,
        do_pos_encoding=POS_ENCODING,
        d_embedding=8,
        # d_embedding=tune.grid_search([8, 16]),
        retrieve_attention_mask=False,
        dropout_p=0.1,
    )

    # OPTIMIZER PARAMETER
    # config["optim_params"] = dict(lr=0.01, amsgrad=True, weight_decay=tune.grid_search([0.1, 0.01]))
    config["optim_params"] = dict(lr=0.01, amsgrad=True, weight_decay=0.1)
    # SCHEDULER PARAMETER
    config["scheduler_params"] = dict(max_lr=0.01, pct_start=0.2)
    #config["scheduler_params"] = dict(max_lr=tune.grid_search([0.01, 0.001]), pct_start=0.2)
    # config["scheduler_params"] = dict(T_0=20, T_mult=1, eta_min=0.001)
    # LOSS
    # config["loss_params"] = dict(gamma=10)
    # TUNE PARAMETER
    config["tune_params"] = dict(
        trial_stopper=True,
        trial_num_results=8,
        trial_grace_period=16,
        trial_tol=0.0,
        experiment_stopper=False,
        exp_top_models=10,
        exp_num_results=64,
        exp_grace_period=0,
        exp_tol=0.0,
    )

    # ---- RAY TUNE SETUP
    search_alg = BasicVariantGenerator()

    stopper = ExperimentAndTrialPlateauStopper(
        metric="score",
        mode="min",
        epochs=config["hyper_params"]["epochs"],
        trial_logfile=os.path.join(logLoc, "logs/trial_log.txt"),
        exp_logfile=os.path.join(logLoc, "logs/early_stopping_log.txt"),
        **config["tune_params"],
    )

    callback = MyCallback(
        mode="min",
        stopper=stopper,
        exp_logfile=os.path.join(logLoc, "logs/experiment_log.txt"),
        exp_top_models=config["tune_params"]["exp_top_models"],
        exp_num_results=config["tune_params"]["exp_num_results"],
        exp_grace_period=config["tune_params"]["exp_grace_period"],
        trial_grace_period=config["tune_params"]["trial_grace_period"],
        exp_tol=config["tune_params"]["exp_tol"],
    )

    tune_params = dict(
        local_dir=logLoc,
        sync_config=tune.SyncConfig(syncer=None),
        scheduler=None,
        search_alg=search_alg,
        stop=stopper,
        callbacks=[callback] if callback is not None else None,
        num_samples=config["hyper_params"]["num_samples"],
        verbose=0,
        metric="score",
        mode="min",
        reuse_actors=False,
        max_failures=8,
        name=trial_name,
        keep_checkpoints_num=16,
        trial_dirname_creator=trial_name_creator,
    )

    save_config(config.copy(), logLoc, add_params={})

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

    # ---- RUN TRAINING
    start_time = time.time()
    trainer = Trainer(
        backend=TorchConfig(backend=BACKEND),
        num_workers=NUM_WORKERS,
        use_gpu=gethostname() in  ["D-1210W11"],
        # resources_per_worker={"CPU": 1} if HOST.startswith("D-1210W17") else {"CPU": 1.0, "GPU": 0.2},
        logdir=logLoc,
    )
    trainable = trainer.to_tune_trainable(train_func)
    analysis = tune.run(
        trainable,
        config=config,
        resume=True if RESUME_TRIAL != "" else False,
        **tune_params,
    )
    save_best_trials(logLoc, n_best_models_to_save=3)
    estimation_time = (time.time() - start_time) / 60
    timed_callout(f" Estimation took {estimation_time:.3f} minutes")

    # ---- RUN Testing
    start_time = time.time()

    best_df = pd.read_pickle(os.path.join(logLoc, "configs.pkl"))

    config_list = best_df.loc[:, [col for col in best_df.columns if "config/" in col]]
    config_list.columns = [c.replace("config/", "") for c in config_list.columns]
    config_list = [x._asdict() for x in config_list.itertuples()]

    trainer = Trainer(
        backend=TorchConfig(backend=BACKEND),
        num_workers=NUM_WORKERS,
        use_gpu= gethostname() in  ["D-1210W11"],
        logdir=logLoc,
    )
    trainer.start()
    for config in config_list:
        # print(f"Working on config {config['hyper_params']['input_parse']}.")
        trainer.run(test_func, config, checkpoint_strategy=None, callbacks=None, checkpoint=None)
    trainer.shutdown()

    estimation_time = (time.time() - start_time) / 60
    timed_callout(f" Testing took {estimation_time:.3f} minutes")

    # cleanup
    ray.shutdown()

# %%

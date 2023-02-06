# %%
from socket import gethostname
import os
import datetime
import time

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
#from attention.loss_functions import MSE_with_mask
from attention.loop import train_epoch, validate_epoch


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
        name="train",
    )
    val_data = ModelDataset(
        filename="../04_results/Momentum_DataSet_TargetOnly=True.pq",
        val_pct=val_pct,
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
if __name__ == "__main__":
    MODEL_LOC = "../05_models"
    MODEL_TYPE = "SingleHeadAttention"

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
        trial_name = MODEL_TYPE + "___" + str(trial_number)
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
        epochs=200,
        trialLoc=os.path.join(logLoc, trial_name),
        val_pct=0.3,
        batch_pct=0.01,
    )

    # MODEL PARAMETER
    config["model_params"] = dict(
        n_lags = 231,
        alpha = 1.0, 
        dropout_p = 0.1,
        retrieve_attention_mask=False,
    )
    # OPTIMIZER PARAMETER
    config["optim_params"] = dict(lr=0.001, amsgrad=True, weight_decay=0.1)
    # SCHEDULER PARAMETER
    config["scheduler_params"] = dict(max_lr=0.001, pct_start=0.2)
    # LOSS
    #config["loss_params"] = dict(gamma=10)
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
        backend=TorchConfig(backend="gloo"),
        num_workers=NUM_WORKERS,
        use_gpu=True if torch.cuda.device_count() > 0 else False,
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
    save_best_trials(logLoc, n_best_models_to_save=1)
    estimation_time = (time.time() - start_time) / 60
    timed_callout(f" Estimation took {estimation_time:.3f} minutes")

    ray.shutdown()

# %%
# %%
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from attention.model import SingleHeadAttention
from attention.utils import timed_callout

from joblib import Parallel, delayed
import os
# %%

class ModelDataset(Dataset):
    """
    ---- PyTorch Dataset
    
    Inputs:
    
    """

    def __init__(
        self,
        val_pct: float,
        name: str,
        seed: int = 1234,
        type = "sinus"
    ):

        """ Initialize Dataset """
        super().__init__()
        N_firms = 1000
        N_returns = 252 * 20    # 20 Years
        sigma_m = 20            # % per year
        mean_m = 6              # % per year
        
        # ----- Simulate data
        # NOTE Assume Market Model 
        np.random.seed(seed)
        
        # Get betas for each stock
        betas = 2 * np.random.random(N_firms) # Uniform in [0,2)

        # Variance inflation (relative to market) for each stock
        var_i = 0.5 + np.random.random(N_firms) # Uniform in [0.5, 1.5)

        # Get market
        loc = (1+mean_m/100)**(1/(252))-1
        scale = np.sqrt(sigma_m/(252*100))
        market = np.random.normal(loc=loc, scale=scale, size=(1,N_returns))

        # Get epsilon
        # Shape B X T
        epsilon_i =  var_i[:,None] * np.random.normal(loc=0, scale=scale, size=(N_firms, N_returns))


        # Calcualte individual stock returns
        r_i = betas[:,None] * market + epsilon_i

        # Transfer to pandas
        data = pd.DataFrame(r_i).stack().to_frame("r")
        data.index.names = ["id", "date"]
        data = data.reset_index()
        
        def data_prep_func(df):
            df = df.set_index("date")

            end = df.shape[0] - 252
            out = []
            for start in np.arange(0,end+1,21):
                tmp = df.iloc[start:start+252].r.values 
                tmp = pd.DataFrame(
                    tmp[None,:],
                    columns = [f"r_{d}" for d in np.arange(-len(tmp),0)],
                    index=[0]
                )
                tmp["id"] = df.id.unique()[0]
                tmp["date"] = start+252
                out.append(tmp)
            
            out = pd.concat(out)
            return out
        
        data = Parallel(os.cpu_count()//2, backend="loky", verbose=True)(
            delayed(data_prep_func)(g) for _,g in data.groupby("id")
        )
        data = pd.concat(data).reset_index(drop=True)

        # Calculate target
        data = data.set_index(["id","date"])
        data = np.log(1+data)     # Log returns
        T_lookback = data.shape[1]
        
        if type == "sinus":
            # Sinus
            multiplier = np.sin(np.arange(T_lookback)*2*np.pi/T_lookback) + 2
            multiplier /= np.sum(multiplier)
        elif type == "exp":
            # Exponential decay
            multiplier = np.exp(0.02*np.arange(T_lookback))
            multiplier /= np.sum(multiplier)
        else:
            raise ValueError("Type msut be in [sinus, exp]")
        
        data["target"] = data.mul(multiplier, axis=1).sum(axis=1)

        # --------------------------------------------------------------
        

        # ---- Data Split
        data = data.reset_index()
        unique_dates = pd.Series(data["date"].unique())
        

        # --- Train/Val/Test split. Random shuffle
        if name == "val":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            data = data[data["date"].isin(val_dates)]
        elif name == "train":
            val_dates = unique_dates.sample(frac=val_pct, random_state=seed)
            train_dates = unique_dates[~unique_dates.isin(val_dates)]
            data = data[data["date"].isin(train_dates)]
        else: # FULL Sample
            timed_callout("Using Full Sample.")

        # --- Get characteristic columns
        self.char_cols = [c for c in data.columns if c not in ["target", "id", "date"]]

        # --- Pytorch needs numpy
        # NOTE in % so that the loss gets bigger ?
        target = (data["target"]).to_numpy().astype("float32")
        chars = (data[self.char_cols] ).to_numpy().astype("float32")
        ids = data["id"].to_numpy().astype("int32")
        dates = data["date"].to_numpy().astype("int64")
 


        # ---- Get evenly divisible number of samples, which is required by PyTorch's DDP
        self.N = data.shape[0]

        # --- Create tensors
        self.target = torch.as_tensor(target)
        self.chars = torch.as_tensor(chars)
        self.ids = torch.as_tensor(ids)
        self.dates = torch.as_tensor(dates)
    

    def __len__(self):
        """ Total lenght of current Dataset"""
        return self.N

    def __getitem__(self, idx):
        """ Get next item """
        
        return (
            self.chars[idx],
            self.target[idx],
            self.ids[idx],
            self.dates[idx],
        )
    
    def retrieve_column_names(self):
        return self.char_cols





class EarlyStopper:
    def __init__(
        self,
        patience: int = 1,
        tol: float = 1.0,  # Deviation in % from minimum validation loss  
        ):

        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.min_val_loss = np.inf
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
        elif validation_loss > ( ( 1 + self.tol/100) * self.min_val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
# %%
def train_epoch(
    dataLoader,
    model,
    loss_fn,
    optimizer,
    epoch: int = 0,
):
    device = torch.device("cpu")
    nobs = len(dataLoader)

    #print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 1])).astype("int")
    print_batches = [nobs-1]


    # --- Set model to train mode
    model.train()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    for batch,(chars, target, _, _) in enumerate(dataLoader):

        # ---1: Send input data to device
        chars = chars.to(device)

        # ---2: Run the model
        prediction, _= model.forward(chars)

        # ---3: Calculate the loss
        loss = loss_fn(prediction,target)

        # ---4: Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # ---5: Update parameters   
        optimizer.step()
        scheduler.step()  # for OneCycle

        # ---- Record statistics
        running_loss += loss.item() * target.numel()
        running_N += target.numel()
        running_mse += ((prediction-target)).pow(2).sum().item()
        running_benchmark += ((target-0)).pow(2).sum().item()

        if batch in print_batches:
            timed_callout(
                f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                + f"Total={running_loss/running_N:.5f} --- "
                + f"R2={1-running_mse/ running_benchmark:.5f}"
            )

        last_lr = scheduler.get_last_lr()[0]
        
    return running_loss/running_N, 1-running_mse/running_benchmark, last_lr

# %%
def val_epoch(
    dataLoader,
    model,
    loss_fn,
    epoch: int = 0,
):
    device = torch.device("cpu")
    nobs = len(dataLoader)

    #print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 1])).astype("int")
    print_batches = [nobs-1]


    # --- Set model to eval mode
    model.eval()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    with torch.no_grad():
        for batch,(chars, target, _, _ ) in enumerate(dataLoader):
            # ---1: Send input data to device
            chars = chars.to(device)

            # ---2: Run the model
            prediction, _ = model.forward(chars)

            # ---3: Calculate the loss
            loss = loss_fn(prediction,target)

            # ---- Record statistics
            running_loss += loss.sum().item() * target.numel()
            running_N += target.numel()
            running_mse += ((prediction-target)).pow(2).sum().item()
            running_benchmark += ((target-0)).pow(2).sum().item()

            if batch in print_batches:
                timed_callout(
                    f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                    + f"Total={running_loss/running_N:.5f} --- "
                    + f"R2={1-running_mse/ running_benchmark:.5f} "
                )
        
    return running_loss/running_N, 1-running_mse/running_benchmark



# %%
def test_epoch(
    dataLoader,
    model,
    loss_fn,
):
    device = torch.device("cpu")
    nobs = len(dataLoader)


    # --- Set model to eval mode
    model.eval()

    # --- Get attention weights
    model.retrieve_attention_mask = True

    # --- Get char_names
    char_names = test_loader.dataset.retrieve_column_names()


    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    final_att = []
    with torch.no_grad():
        for batch,(chars, target, ids, dates ) in enumerate(dataLoader):
            # ---1: Send input data to device
            chars = chars.to(device)

            # ---2: Run the model
            prediction, att = model.forward(chars)

            # ---3: Calculate the loss
            loss = loss_fn(prediction,target)

            # Attention
            tmp = pd.DataFrame(att.detach().cpu().numpy(), columns=char_names)
            tmp["date"] = dates.detach().cpu().numpy()
            tmp["id"] = ids.detach().cpu().numpy()
            final_att.append(tmp)

            # ---- Record statistics
            running_loss += loss.sum().item() * target.numel()
            running_N += target.numel()
            running_mse += ((prediction-target)).pow(2).sum().item()
            running_benchmark += ((target-0)).pow(2).sum().item()

    final_att = pd.concat(final_att)

    timed_callout(f"Testing done: R2={1-running_mse/ running_benchmark:.5f}--- ")

        
    return final_att


# %%
TYPE = "sinus"


config = {}
# MODEL PARAMETER
config["model_params"] = dict(
    n_lags=252,
    alpha=1.0,
    do_pos_encoding=True,
    d_embedding=8,
    # d_embedding=tune.grid_search([8, 16]),
    retrieve_attention_mask=False,
    dropout_p=0.1,
)


config["optim_params"] = dict(lr=0.1, amsgrad=True, weight_decay=0.1)
config["scheduler_params"] = dict(max_lr=0.1, pct_start=0.2)

max_epochs = 100 

train_data = ModelDataset(val_pct=0.3, name="train", type= TYPE)
val_data = ModelDataset(val_pct=0.3, name="val", type= TYPE)


#batch_size = 8**5
batch_size = (round(len(train_data)*0.01) // 8 ) * 8 + 8

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

# %%

loss_fn = nn.MSELoss()

model = SingleHeadAttention(**config["model_params"])
# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(),**config["optim_params"])
# SCHEDULER
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    **config["scheduler_params"],
    epochs=max_epochs,
    steps_per_epoch=(len(train_data) // batch_size // 1) + 1,
)

# %%
train_loss = []
val_loss = []
train_r2 = []
val_r2 = []
lr = []

early_stopper = EarlyStopper(patience=5, tol=3.0)
for epoch in range(1,max_epochs+1):

   # ---- Training 
    tmp_loss, tmp_r2, last_lr = train_epoch(
        dataLoader=train_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epoch=epoch)
    train_loss.append(tmp_loss)
    train_r2.append(tmp_r2)
    lr.append(last_lr)

    # ---- Validation
    tmp_val_loss, tmp_r2= val_epoch(
        dataLoader=val_loader,
        model=model,
        loss_fn=loss_fn,
        epoch=epoch)
    val_loss.append(tmp_val_loss)
    val_r2.append(tmp_r2)


    # ---- Plot
    if epoch%10==0:
        fig,ax=plt.subplots(3,1,sharex=True)
        ax[0].plot(np.arange(len(train_loss)),train_loss,"-ob",label="Training")
        ax[0].plot(np.arange(len(val_loss)),val_loss,"-or",label="Validation")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("")

        ax[1].plot(np.arange(len(train_r2)),train_r2,"-ob",label="Training")
        ax[1].plot(np.arange(len(val_r2)),val_r2,"-or",label="Validation")
        ax[1].set_ylabel(r"$R^2$")
        ax[1].set_xlabel("")

        ax[2].plot(np.arange(len(lr)),lr,"-b",label="Learning Rate")
        ax[2].set_ylabel(r"LR")
        ax[2].set_xlabel("Epoch")
        plt.pause(0.5)
    
    if early_stopper.early_stop(tmp_val_loss):
        fig,ax=plt.subplots(3,1,sharex=True)
        ax[0].plot(np.arange(len(train_loss)),train_loss,"-ob",label="Training")
        ax[0].plot(np.arange(len(val_loss)),val_loss,"-or",label="Validation")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("")

        ax[1].plot(np.arange(len(train_r2)),train_r2,"-ob",label="Training")
        ax[1].plot(np.arange(len(val_r2)),val_r2,"-or",label="Validation")
        ax[1].set_ylabel(r"$R^2$")
        ax[1].set_xlabel("")

        ax[2].plot(np.arange(len(lr)),lr,"-b",label="Learning Rate")
        ax[2].set_ylabel(r"LR")
        ax[2].set_xlabel("Epoch")
        plt.pause(0.5)
        
        break

# ----  Testing
test_data = ModelDataset(val_pct=0.3, name="test", type= TYPE)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
att = test_epoch(dataLoader=test_loader, model=model, loss_fn=loss_fn)

# Saving
att.to_parquet(f"../04_results/Sim={TYPE}_AttentionWeights.pq")

# %%
# --- Plotting
plot_data = pd.read_parquet(f"../04_results/Sim={TYPE}_AttentionWeights.pq")
plot_data = plot_data.set_index(["date", "id"]).mean()


fig, ax = plt.subplots()
x = np.arange(-plot_data.shape[0],0)


if TYPE=="sinus":
    # Sinus
    true_val = np.sin(np.arange(len(x))*2*np.pi/len(x)) + 2
    true_val /= true_val.sum()
elif TYPE=="exp":
    # Exponential
    true_val = np.exp(0.02*np.arange(len(x)))
    true_val /= np.sum(true_val)

ax.plot(x, plot_data.values, ls="", marker=".", color="b", label="Simulation")
ax.plot(x, true_val, color = "k", )
ax.axhline(plot_data.mean(), ls="--", color="k")

ax.set_xlabel("Lag (Days)")
ax.set_ylabel("Attention weight")
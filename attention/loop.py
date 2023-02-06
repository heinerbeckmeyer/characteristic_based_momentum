import pandas as pd
import torch

from attention.utils import timed_callout


# %%
def train_epoch(
    dataLoader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    epoch: int = 0,
):

    nobs = len(dataLoader)

    # print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 0.99])).astype("int")
    print_batches = [nobs - 1]

    # --- Set model to train mode
    model.train()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    for batch, (chars, target, _, _, _, _) in enumerate(dataLoader):
        # ---1: Run the model
        prediction, _ = model.forward(chars)

        # ---2: Calculate the loss
        loss = loss_fn(prediction, target)

        # ---3: Backpropagation
        optimizer.zero_grad(set_to_none=True)  # May speed ip comp.
        loss.backward()

        # ---4: Update parameters
        optimizer.step()
        scheduler.step()  # for OneCycle

        # ---- Record statistics
        running_loss += loss.item() * target.numel()
        running_N += target.numel()

        running_mse += ((prediction - target)).pow(2).sum().item()
        # running_benchmark += ((target - target.mean())).pow(2).sum().item()
        running_benchmark += ((target - 0)).pow(2).sum().item()

        if batch in print_batches:
            timed_callout(
                f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                + f"Total={running_loss/running_N:.5f} --- "
                + f"R2={1-running_mse/ running_benchmark:.5f}"
            )
            
        
        

    return running_loss / running_N, 1 - running_mse / running_benchmark


# %%
def validate_epoch(
    dataLoader,
    model,
    loss_fn,
    epoch: int = 0,
):

    nobs = len(dataLoader)

    # print_batches = (nobs * np.array([0, 0.25, 0.5, 0.75, 0.99])).astype("int")
    print_batches = [nobs - 1]

    # --- Set model to eval mode
    model.eval()

    running_loss = 0.0
    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    with torch.no_grad():
        for batch, (chars, target, _, _, _, _) in enumerate(dataLoader):
            # For Testing make easy target:
            #target = torch.mean(chars[:,-21:], axis=1).clone()

            # ---1: Run the model
            prediction, _ = model.forward(chars)

            # ---2: Calculate the loss
            loss = loss_fn(prediction, target)

            # ---- Record statistics
            running_loss += loss.item() * target.numel()
            running_N += target.numel()
            running_mse += ((prediction - target)).pow(2).sum().item()
            # running_benchmark += ((target - target.mean())).pow(2).sum().item()
            running_benchmark += ((target - 0)).pow(2).sum().item()

            if batch in print_batches:
                timed_callout(
                    f"Epoch {epoch}, {batch}/{nobs}: Loss={loss.item():.5f} --- "
                    + f"Total={running_loss/running_N:.5f} --- "
                    + f"R2={1-running_mse/ running_benchmark:.5f}"
                )

    return running_loss / running_N, 1 - running_mse / running_benchmark


# %%
def test_epoch(dataLoader, model):
    # nobs = len(dataLoader)

    # --- Set model to eval mode
    model.eval()

    running_N = 0
    running_mse = 0.0
    running_benchmark = 0.0

    char_names = dataLoader._dataloader.dataset.retrieve_column_names()

    final_pred = []
    final_att = []
    with torch.no_grad():
        for _, (chars, target, target_unchanged, ids, dates, benchmark) in enumerate(dataLoader):
            # For Testing make easy target:
            #target = torch.mean(chars[:,-21:], axis=1).clone()
            # ---1: Run the model
            prediction, attention = model.forward(chars)

            # --- Create Dataframe
            # Prediction
            out = pd.DataFrame(prediction.detach().cpu().numpy(), columns=["prediction"])
            out["date"] = dates.detach().cpu().numpy().astype("datetime64[ns]")
            out["id"] = ids.detach().cpu().numpy()
            out["target"] = target_unchanged.detach().cpu().numpy()
            out["benchmark"] = benchmark.detach().cpu().numpy()
            final_pred.append(out)

            # Attention
            tmp = pd.DataFrame(attention.detach().cpu().numpy(), columns=char_names)
            tmp["date"] = dates.detach().cpu().numpy().astype("datetime64[ns]")
            tmp["id"] = ids.detach().cpu().numpy()
            final_att.append(tmp)

            # ---- Record statistics
            running_N += target.numel()
            running_mse += ((prediction - target)).pow(2).sum().item()
            # running_benchmark += ((target - target.mean())).pow(2).sum().item()
            running_benchmark += ((target - 0)).pow(2).sum().item()

    final_pred = pd.concat(final_pred)
    final_att = pd.concat(final_att)

    timed_callout(f"Testing done: R2={1-running_mse/ running_benchmark:.5f}--- ")

    return final_pred, final_att

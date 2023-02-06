# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%


class CrossEntropyLoss_withMask:
    """
    Pytorch CrossEntropyLoss with obfuscation mask
    """

    def __init__(self):
        self._name = "ce_loss"

    def __call__(self, prediction, target, mask):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = torch.sum(loss_fn(prediction.transpose(2, 1), target.long()) * mask) / mask.sum()
        return loss


class R2Loss:
    """
    Pytorch Rsquared-Loss
    """

    def __init__(self):
        self._name = "r2_loss"

    def __call__(self, prediction, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target-target_mean) ** 2)
        ss_res = torch.sum((target-prediction) ** 2)
        r2_loss = 1 - ss_res / ss_tot

        return r2_loss 

# %%
class FocalLoss_withMask(nn.modules.loss._WeightedLoss):
    """
    FocalLoss with obfuscation mask
    """

    def __init__(self, weight: float = None, gamma: int = 2, reduction: str = "none"):
        super().__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight
        self._name = "focal_loss"

    def __call__(self, prediction, target, mask):
        ce_loss = F.cross_entropy(
            prediction.transpose(2, 1), target.long(), reduction=self.reduction, weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        focal_loss = torch.sum(focal_loss * mask) / mask.sum()

        return focal_loss


# %%
class MSE_with_mask:
    """ MSE using obfuscation mask. """

    def __init__(self):
        self._name = "mse"

    def __call__(self, prediction, target, mask):
        loss = torch.sum(((prediction - target) * mask) ** 2) / mask.sum()
        # NOTE: we do not need to scale with target standard deviation, as all our input features
        # are normalized to [-1, 1]
        return loss
"""Loss."""

import torch
import torch.nn as nn

class Loss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(Loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, classes, wloss, device):        
        if classes > 2:
            ce_loss = nn.functional.cross_entropy(
                pred, target, reduction=self.reduction
            )
            
        elif classes == 2:
            
            target = target.unsqueeze(-1).float()
            if wloss:
                #print("\n", "Using weighted version of the loss", "\n")
                weights = torch.where(
                    target == 1,
                    torch.tensor([0.7]).to(device),
                    torch.tensor([0.3]).to(device),
                )
                ce_loss = nn.functional.binary_cross_entropy(
                    torch.sigmoid(pred),
                    target,
                    reduction=self.reduction,
                    weight=weights,
                )
            else:
                ce_loss = nn.functional.binary_cross_entropy(
                    torch.sigmoid(pred),
                    target,
                    reduction=self.reduction,
                )
        return ce_loss.requires_grad_()

"""
## NOTICE ##
THIS CODE IS TAKEN FROM:
https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
"""
from torch import nn
from torch.nn import functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=3):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
       # inputs = inputs.view(-1)
       # targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(axis=[2, 3])
        FP = ((1 - targets) * inputs).sum(axis=[2, 3])
        FN = (targets * (1 - inputs)).sum(axis=[2, 3])

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky_batch = (1 - Tversky) ** gamma

        return FocalTversky_batch.squeeze(1)
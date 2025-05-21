# -- coding : uft-8 --
# Author : Wang Han 
# Southeast University
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import one_hot


def focal_loss(input_values, gamma, alpha=0.25):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    # loss = (1 - p) ** gamma * input_values
    loss = (1 - p) ** gamma * input_values * alpha * 10
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, to_one_hot=True, weight=None, gamma=2, alpha=0.25, num_classes=2):
        super(FocalLoss, self).__init__()
        if weight is None:
            weight = [1, 1]
        assert gamma >= 0
        self.gamma = gamma
        self.weight = torch.tensor(weight)
        self.alpha = alpha
        self.to_one_hot = to_one_hot
        self.num_classes = num_classes

    def forward(self, input, target):
        target = target.float()

        if self.to_one_hot:
            target = one_hot(target, num_classes=self.num_classes)
        else:
            target = torch.tensor(target)

        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight, label_smoothing=0.2),
                          self.gamma, self.alpha)


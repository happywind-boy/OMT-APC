import time
import torch
import numpy as np
from src.metric import dice
import torch.nn.functional as F


def rot90(inputs, inverse=False):
    if inverse:
        outputs = torch.rot90(inputs.squeeze(0), k=-1, dims=(1,2))

    else:
        outputs = torch.rot90(inputs.squeeze(0), k=1, dims=(1,2))

    return outputs.unsqueeze(0)


def fliplr(inputs, inverse=False):
    inputs = inputs.squeeze(0)
    outputs = torch.zeros_like(inputs)
    
    for i in range(inputs.shape[0]):
        outputs[i, :, :, :] = torch.fliplr(inputs[i, :, :, :])

    return outputs.unsqueeze(0)


def flipud(inputs, inverse=False):
    inputs = inputs.squeeze(0)
    outputs = torch.zeros_like(inputs)
    
    for i in range(inputs.shape[0]):
        outputs[i, :, :, :] = torch.flipud(inputs[i, :, :, :])

    return outputs.unsqueeze(0)


def fliplr_rot90(inputs, inverse=False):
    if inverse:
        tmp = fliplr(inputs, inverse=True)
        outputs = rot90(tmp, inverse=True)

    else:
        outputs = fliplr(rot90(inputs))

    return outputs


def flipud_rot90(inputs, inverse=False):
    if inverse:
        tmp = flipud(inputs, inverse=True)
        outputs = rot90(tmp, inverse=True)

    else:
        outputs = flipud(rot90(inputs))

    return outputs




import time
import torch
import numpy as np
from src.metric import dice
import torch.nn.functional as F


def identity(inputs, inverse=False):
    return inputs


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


def transform_predicted(model, T, image):
    invT = lambda x: T(x, inverse=True)

    # return F.softmax(invT(model(T(image))), dim=1)
    return invT(model(T(image)))


def compute_dice_sim(x, y):
    """Compute the accuracy
    Args:
        x (torch.FloatTensor): predictions with shape (1, C, n, n, n).
        y (torch.FloatTensor): ground truth with shape (1, C, n, n, n).

    Returns:
        similarity (torch.FloatTensor): dice similarity
    """
    x = (x>=0.5).float() #torch.argmax(x, dim=1)
    y = (y>=0.5).float() #torch.argmax(y, dim=1)

    return dice(x, y).mean().item()


def compute_dice_sim_matrix(pred_seq):
    n = len(pred_seq)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            A[i, j] = compute_dice_sim(pred_seq[i], pred_seq[j])

    dice_sim_matrix = A + A.T + np.diag([1]*n)

    return dice_sim_matrix


def print_matrix(M):
    print("\n")
    m, n = np.shape(M)
    for i in range(m):
        for j in range(n):
            print(round(M[i][j], 2), end='\t')

        print('\n')

    return


def compute_first_ev(matrix):
    ew, ev = np.linalg.eig(matrix)
    max_index = np.argmax(ew)
    first_ev = ev[:, max_index]

    return first_ev


def exp_norm(vec, e=100):
    exp = e**vec

    return exp / np.sum(exp)


def compute_coeff(vector):
    # coeff = exp_norm(np.abs(vector))
    vec = np.abs(vector)
    coeff = vec / np.sum(vec)

    return coeff


def compute_weighted_sum(preds, coeff):
    preds = torch.stack([c*p for (c, p) in zip(coeff, preds)])
    pred = torch.sum(preds, dim=0)

    return pred


def adapative_self_ensemble(model, image):
    """Do the adapative self-ensemble
    Args:
        model
        image (torch.FloatTensor): with shape (1, C, n, n, n).

    Returns:
        prediction (torch.FloatTensor): with shape (1, 1, n, n, n).
    """
    pred_seq = [
        model(image),
        transform_predicted(model, rot90, image),
        transform_predicted(model, fliplr, image),
        transform_predicted(model, flipud, image),
        transform_predicted(model, fliplr_rot90, image),
        transform_predicted(model, flipud_rot90, image)
    ]
    dice_sim_matrix = compute_dice_sim_matrix(pred_seq)
    first_ev = compute_first_ev(dice_sim_matrix)
    coeff = compute_coeff(first_ev)
    pred = compute_weighted_sum(pred_seq, coeff)

    return pred


def self_ensemble(model, image):
    """Do the self-ensemble
    Args:
        model
        image (torch.FloatTensor): with shape (1, C, n, n, n).

    Returns:
        prediction (torch.FloatTensor): with shape (1, 1, n, n, n).
    """
    pred_seq = [
        model(image),
        transform_predicted(model, rot90, image),
        transform_predicted(model, fliplr, image),
        transform_predicted(model, flipud, image),
        transform_predicted(model, fliplr_rot90, image),
        transform_predicted(model, flipud_rot90, image)
    ]
    coeff = [int(compute_dice_sim(pred_seq[0], pred_seq[i])>0.8) for i in range(len(pred_seq))]
    pred = compute_weighted_sum(pred_seq, coeff)

    return pred


def adapative_ensemble(models, image):
    """Do the ensemble
    Args:
        model
        image (torch.FloatTensor): with shape (1, C, n, n, n).

    Returns:
        prediction (torch.FloatTensor): with shape (1, 1, n, n, n).
    """
    pred_seq = [m(image) for m in models]
    dice_sim_matrix = compute_dice_sim_matrix(pred_seq)
    first_ev = compute_first_ev(dice_sim_matrix)
    coeff = compute_coeff(first_ev)
    pred = compute_weighted_sum(pred_seq, coeff)

    print_matrix(dice_sim_matrix)

    return pred


def ensembles(models, image):
    pred_seq = []

    for model in models:
        pred_seq += [
            model(image),
            transform_predicted(model, rot90, image),
            transform_predicted(model, fliplr, image),
            transform_predicted(model, flipud, image),
            transform_predicted(model, fliplr_rot90, image),
            transform_predicted(model, flipud_rot90, image)
        ]
    #N = len(pred_seq)
    coeff = [int(compute_dice_sim(pred_seq[0], pred_seq[i])>=0) for i in range(len(pred_seq))]
    N = np.sum(coeff)
    pred = compute_weighted_sum(pred_seq, coeff) / N
    #dice_sim_matrix = compute_dice_sim_matrix(pred_seq)
    #first_ev = compute_first_ev(dice_sim_matrix)
    #coeff = compute_coeff(first_ev)
    #pred = compute_weighted_sum(pred_seq, coeff)

    #print_matrix(dice_sim_matrix)
    #print(coeff)

    return pred


def new_ensembles(models, image):
    pred_seq = []

    for model in models:
        pred_seq += [
            model(image),
            transform_predicted(model, rot90, image),
            transform_predicted(model, fliplr, image),
            transform_predicted(model, flipud, image),
            transform_predicted(model, fliplr_rot90, image),
            transform_predicted(model, flipud_rot90, image)
        ]
    preds = torch.stack(pred_seq)
    pred = torch.mean(preds, dim=0)
    nonzero_num = (pred[:, -1] > 0.5).int().sum()
    #print(nonzero_num)
    if nonzero_num < 120:
        print('This is empty.\n')
        return pred

    else:
        dice_sim_matrix = compute_dice_sim_matrix(pred_seq)
        first_ev = compute_first_ev(dice_sim_matrix)
        coeff = compute_coeff(first_ev)
        pred = compute_weighted_sum(pred_seq, coeff)

        #print_matrix(dice_sim_matrix)
        #print(coeff)

    return pred


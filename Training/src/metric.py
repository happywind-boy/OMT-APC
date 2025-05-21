import numpy as np
import torch
import monai
from monai.networks.utils import one_hot
from monai.metrics import compute_hausdorff_distance
import torchmetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice(A, B, epsilon=1e-5):
    """Compute the Dice Similarity Coefficient (DSC)
    Args:
        A (torch.FloatTensor): predictions with shape (B, C, n, n, n).
        B (torch.FloatTensor): ground truth with shape (B, C, n, n, n).
        epsilon (float): smooth epsilon

    Returns:
        score (torch.FloatTensor): mean of dice
    """
    bs = A.shape[0]
    A = A.reshape(bs, -1)  # (B, C, n, n, n) -> (B, C*n*n*n)
    B = B.reshape(bs, -1)  # (B, C, n, n, n) -> (B, C*n*n*n)
    int_sum = torch.sum(A * B, dim=1)
    deno_sum = torch.sum(A + B, dim=1)
    score = (2 * int_sum + epsilon) / (deno_sum + epsilon)

    return score.mean()


def metrics(PD, GT):
    PD = torch.tensor(PD)
    GT = torch.tensor(GT)
    N = PD.shape[0]

    TP = torch.sum(PD * GT)
    FN = torch.sum(GT) - TP
    FP = torch.sum(PD) - TP
    TN = N - (TP + FN + FP)

    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    F1 = (2 * ppv * sensitivity) / (ppv + sensitivity)
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return sensitivity, specificity, ppv, npv, F1, MCC

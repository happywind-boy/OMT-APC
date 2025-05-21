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


def compute_accuracy(pred, label):
    acc_compute = torchmetrics.Accuracy(task='binary').to(device)
    acc = acc_compute(pred, label)

    return acc


def compute_auc(pred, label):
    auc_compute = torchmetrics.AUROC(task='binary').to(device)
    auc = auc_compute(pred, label)
    return auc


def compute_recall(pred, label):
    recall_compute = torchmetrics.Recall(task='binary').to(device)
    recall = recall_compute(pred, label)
    return recall


def compute_precision(pred, label):
    precision_compute = torchmetrics.Precision(task='binary').to(device)
    precision = precision_compute(pred, label)
    return precision


def compute_acc(pred, label, act=None):
    """Compute the accuracy
    Args:
        preds (torch.FloatTensor): predictions with shape (B, C, n, n, n).
        masks (torch.FloatTensor): ground truth with shape (B, C, n, n, n).
        prob_type (bool): whether the prediction is type of probabilty?

    Returns:
        score (torch.FloatTensor): mean of scores
    """
    if act == 'softmax':
        pred = torch.argmax(pred, dim=1)

    elif act == 'sigmoid':
        pred = pred >= 0.5

    return dice(pred, label)


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


class Metrics():
    def __init__(self):
        self.eps = 1e-5
        # self.metrics_name = []

    def confusion_matrix(self, PD, GT):
        bs = PD.shape[0]
        PD = PD.reshape(bs, -1)
        GT = GT.reshape(bs, -1)
        N = PD.shape[1]

        TP = torch.sum(PD * GT, dim=1)
        FN = torch.sum(GT, dim=1) - TP
        FP = torch.sum(PD, dim=1) - TP
        TN = N - (TP + FN + FP)

        return TP, FN, FP, TN

    def precision(self, TP, FN, FP, TN):
        return (TP + self.eps) / (TP + FP + self.eps)

    def recall(self, TP, FN, FP, TN):
        return (TP + self.eps) / (TP + FN + self.eps)

    def dice(self, TP, FN, FP, TN):
        return (2 * TP + self.eps) / (2 * TP + FP + FN + self.eps)

    def specificity(self, TP, FN, FP, TN):
        return (TN + self.eps) / (TN + FP + self.eps)

    def hausdorff_distance(self, PD, GT):
        PD = one_hot(PD.unsqueeze(1), 2)
        GT = one_hot(GT, 2)
        hausdorff = compute_hausdorff_distance(PD, GT)

        return hausdorff

    def compute_metrics(self, PD, GT, tensor_type):
        cm = self.confusion_matrix(PD, GT)
        metrics = {
            f'{tensor_type}_dice': f"{self.dice(*cm).item():.5f}",
            f'{tensor_type}_precision': f"{self.precision(*cm).item():.5f}",
            f'{tensor_type}_recall': f"{self.recall(*cm).item():.5f}",
            f'{tensor_type}_specificity': f"{self.specificity(*cm).item():.5f}",
            f'{tensor_type}_hausdorff': f"{self.hausdorff_distance(PD, GT).item():.5f}"
        }

        return metrics

# -- coding : uft-8 --
# Author : Wang Han 
# Southeast University
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import one_hot


class BinaryVSLoss(nn.Module):

    def __init__(self, iota_pos=0.0, iota_neg=0.0, Delta_pos=1.0, Delta_neg=1.0, weight=None):
        super(BinaryVSLoss, self).__init__()
        iota_list = torch.tensor([iota_neg, iota_pos])
        Delta_list = torch.tensor([Delta_neg, Delta_pos])

        self.iota_list = torch.Tensor(iota_list)
        self.Delta_list = torch.Tensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros((x.shape[0], 2), dtype=torch.uint8)
        index_float = index.type(torch.Tensor)
        index_float.scatter_(1, target.long(), 1)

        batch_iota = torch.matmul(self.iota_list, index_float.t())
        batch_Delta = torch.matmul(self.Delta_list, index_float.t())

        batch_iota = batch_iota.view((-1, 1))
        batch_Delta = batch_Delta.view((-1, 1))

        output = x * batch_Delta - batch_iota
        output = output.long()

        return F.binary_cross_entropy_with_logits(30 * output, target, weight=self.weight)


class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.Tensor(iota_list)
        self.Delta_list = torch.Tensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.Tensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.Tensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target, weight=self.weight)


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


if __name__ == '__main__':
    lab = torch.tensor([1])
    out1 = torch.tensor([[0.8490, 0.1510]])
    out2 = torch.tensor([[0.4510, 0.5490]])
    Loss1 = FocalLoss(weight=[1, 1.2])
    Loss2 = FocalLoss(weight=[1, 1.1])
    Loss3 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.1]))
    Loss4 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]))
    l1 = Loss1(out1, lab)
    l2 = Loss2(out1, lab)
    l3 = F.cross_entropy(out1, lab)
    l4 = F.cross_entropy(out1, lab, reduction='none', weight=torch.tensor([1, 1.2]))
    print(l1, l2, l3, l4)

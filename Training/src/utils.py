import os
import json
import glob
import monai
import torch
import random
import numpy as np
import nibabel as nib
from datetime import datetime


def set_radom_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return


def init_model(models, ckpts, device):
    for m, c in zip(models, ckpts):
        weight = torch.load(c, map_location=device, weights_only=True)
        m.load_state_dict(weight)
        m.to(device)
        m.eval()

    return models


def get_time():
    return datetime.today().strftime('%m-%d-%H-%M-%S')


def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(vars(obj), fp, indent=4)

    return


def load_json(path):
    with open(path, 'r') as fp:
        obj = json.load(fp)

    return obj


def save_topk_ckpt(model, weight_path, save_name, topk=5):
    model.save(os.path.join(weight_path, save_name))
    weight_list = sorted(
        glob.glob(os.path.join(weight_path, '*.pth')),
        key=lambda x: float(x[-11:-4]), reverse=True)

    # remove the last checkpoint except initial weight
    if len(weight_list) > topk:
        os.remove(weight_list[-2])

    return


def get_topk_ckpt(weight_path, topk=3):
    weight_list = sorted(
        glob.glob(os.path.join(weight_path, 'weight', '*.pth')),
        key=lambda x: float(x[-16:-11]), reverse=True)
    # print(os.path.join(weight_path, 'weight', '*.pth'))

    topk_ckpt = weight_list[:topk]

    return topk_ckpt


def print_dict(**inputs):
    print(', '.join(f"{k}: {v}" for k, v in zip(inputs.keys(), inputs.values())))

    return


def save_tensor(tensor, path, affine=np.eye(4)):
    nib.save(nib.Nifti1Image(tensor, affine), path)

    return


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.total = 0
        self.count = 0

        return

    def update(self, value, batch_size=1):
        self.value = value
        self.count += batch_size
        self.total += value * batch_size
        self.avg = self.total / self.count

        return


def initial_record():
    record = {'Loss': AverageMeter(),
              'ACC': AverageMeter(),
              'AUC': AverageMeter(),
              'Precision': AverageMeter(),
              'Recall': AverageMeter()}

    return record


def init_record(mode='single'):
    if mode == 'single':
        record = {'CLoss': AverageMeter(),
                  'CAcc': AverageMeter(),
                  'BLoss': AverageMeter(),
                  'BAcc': AverageMeter()}

    elif mode == 'multiple':
        record = {'CLoss': AverageMeter(),
                  'wt_CAcc': AverageMeter(),
                  'tc_CAcc': AverageMeter(),
                  'et_CAcc': AverageMeter(),
                  'wt_BAcc': AverageMeter(),
                  'tc_BAcc': AverageMeter(),
                  'et_BAcc': AverageMeter()}

    elif mode == 'crop':
        record = {'BLoss': AverageMeter(),
                  'BAcc': AverageMeter()}

    elif mode == 'multi_crop':
        record = {'BLoss': AverageMeter(),
                  'wt_BAcc': AverageMeter(),
                  'tc_BAcc': AverageMeter(),
                  'et_BAcc': AverageMeter()}

    return record


def to_binary(tensor):
    channel = tensor.shape[1]
    if channel == 1:
        return (tensor >= 0.5).int()

    elif channel == 2:
        return torch.argmax(tensor, dim=1)

    else:
        print('Shape is error.')


def print_list(lis):
    for e in lis:
        print(e)

    return

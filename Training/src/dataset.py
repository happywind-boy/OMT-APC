import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import (Compose, AddChanneld, ToTensord, RandRotate90, RandFlipd,
                              RandCropByPosNegLabeld, CropForegroundd, EnsureTyped, RandSpatialCropSamplesd)
from AutoClassification.src.transforms import *
from sklearn import preprocessing


def LoadImg1(data):
    nifti = nib.load(data)
    data = nifti.get_fdata()

    return data


def norm(img):
    mean = np.mean(img)
    std = np.std(img)
    z = (img - mean) / std
    rmax, rmin = 5, -5
    z[z > rmax] = rmax
    z[z < rmin] = rmin
    img = (z - rmin) / (rmax - rmin)
    return img


def HLGG(grade):
    if grade > 3:
        HLGrade = 1
    else:
        HLGrade = 0
    return HLGrade
  

class UCSFomtDataset(Dataset):
    def __init__(self, data_file, idx_list, opt, transform=None):
        super(UCSFomtDataset, self).__init__()
        self.data_file = data_file
        self.idx_list = idx_list
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        filename = self.idx_list[i]
        f = pd.read_csv(self.opt.label_file[0])
        label = f[f['ID'] == filename]['IDH'].values

        Img = LoadImg1(os.path.join(self.data_file, filename, f'{filename}.nii.gz'))
        t1ce = Img[:, :, :, 0]
        t1ce = norm(t1ce)
        t2 = Img[:, :, :, 1]
        t2 = norm(t2)
        mask = LoadImg1(os.path.join(self.data_file, filename, f'{filename}_seg.nii.gz'))
        image = np.array([t1ce, t2, mask])

        fe = pd.read_csv(self.opt.feature_file[0])
        feature1 = fe[f['ID'] == filename].values[0][2:14]
        feature1 = feature1.astype(float).reshape(-1, 1)
        zscore = preprocessing.StandardScaler()
      
        # normalization
        feature1 = zscore.fit_transform(feature1)
        feature1 = feature1.flatten()

        ft = pd.read_csv(self.opt.tensor_feature_file[0])
        feature2 = ft[f['ID'] == filename].values[0][16:]
        feature2 = feature2.astype(float).reshape(-1, 1)
        feature2 = np.array(feature2.flatten())

        data = {
            'ID': filename,
            'image': image,
            'label': label,
            'feature1': feature1,
            'feature2': feature2
        }

        if self.transform is not None:
            data = self.transform(data)
        # data = {datas.keys[i],datas.values[i]}   
        # print(len(data))
        return data


# ================== dataset =================== #
def get_train_val_test_omt_dataset(train_id, val_id, test_id, data_file, opt):
    train_transform = Compose([
        ToTensord(keys=['image', 'feature1', 'feature2'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64),
        RandRot90(keys=['image'], p=opt.prob_rot90),
        RandFliplr(keys=['image'], p=opt.prob_fliplr),
        RandFlipud(keys=['image'], p=opt.prob_flipud),
        RandomGaussianNoise(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        ToTensord(keys=['image', 'feature1', 'feature2'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64)
    ])

    train_set = UCSFomtDataset(data_file[0], train_id, opt, train_transform)
    val_set = UCSFomtDataset(data_file[0], val_id, opt, val_transforms)
    test_set = UCSFomtDataset(data_file[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set

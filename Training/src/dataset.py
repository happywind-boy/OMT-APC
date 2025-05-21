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


class UCSFDataset(Dataset):
    def __init__(self, data_file, idx_list, opt, transform=None):
        super(UCSFDataset, self).__init__()
        self.data_file = data_file
        self.idx_list = idx_list
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        # omt_file = random.choice(self.data_file)
        # ID = int(self.idx_list[i])
        # filename = f'{self.opt.dataset}_{ID:0>5}'

        filename = self.idx_list[i]
        f = pd.read_csv(self.opt.label_file[0])
        label = f[f['ID'] == filename]['IDH'].values

        t1 = LoadImg1(os.path.join(self.data_file, filename, filename + '_T1_crop.nii.gz'))
        t1 = norm(t1)
        t1ce = LoadImg1(os.path.join(self.data_file, filename, filename + '_T1gad_crop.nii.gz'))
        t2 = LoadImg1(os.path.join(self.data_file, filename, filename + '_T2_crop.nii.gz'))
        t1ce = norm(t1ce)
        t2 = norm(t2)
        flair = LoadImg1(os.path.join(self.data_file, filename, filename + '_FLAIR_crop.nii.gz'))
        flair = norm(flair)
        mask = LoadImg1(os.path.join(self.data_file, filename, filename + '_WT_crop.nii.gz'))
        mask = norm(mask)
        # mask[mask>0] = 1

        image = np.array([t1ce, t2, mask])

        fe = pd.read_csv(self.opt.feature_file[0])
        feature1 = fe[f['ID'] == filename].values[0][2:]
        feature1 = feature1.astype(float).reshape(-1, 1)
        zscore = preprocessing.StandardScaler()
        # 标准化处理
        feature1 = zscore.fit_transform(feature1)
        feature1 = feature1.flatten()

        ft = pd.read_csv(self.opt.tensor_feature_file[0])
        feature2 = ft[f['ID'] == filename].values[0][16:]
        feature2 = feature2.astype(float).reshape(-1, 1)
        feature2 = np.array(feature2.flatten())

        data = {
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
        # omt_file = random.choice(self.data_file)
        # ID = int(self.idx_list[i])
        # filename = f'{self.opt.dataset}_{ID:0>5}'

        filename = self.idx_list[i]
        f = pd.read_csv(self.opt.label_file[0])
        label = f[f['ID'] == filename]['IDH'].values

        Img = LoadImg1(os.path.join(self.data_file, filename, f'{filename}.nii.gz'))
        t1ce = Img[:, :, :, 0]
        t1ce = norm(t1ce)
        t2 = Img[:, :, :, 1]
        t2 = norm(t2)
        mask = LoadImg1(os.path.join(self.data_file, filename, f'{filename}_seg.nii.gz'))
        # mask = norm(mask)
        image = np.array([t1ce, t2, mask])

        fe = pd.read_csv(self.opt.feature_file[0])
        feature1 = fe[f['ID'] == filename].values[0][2:14]
        feature1 = feature1.astype(float).reshape(-1, 1)
        zscore = preprocessing.StandardScaler()
        # 标准化处理
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


class omt1p19qDataset(Dataset):
    def __init__(self, file_ls, idx_list, opt, transform=None):
        super(omt1p19qDataset, self).__init__()
        self.file_list = file_ls
        self.idx_list = idx_list
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        # omt_file = random.choice(self.data_file)
        # ID = int(self.idx_list[i])
        # filename = f'{self.opt.dataset}_{ID:0>5}'

        filename = self.idx_list[i]
        root = self.file_list
        f = pd.read_csv(self.opt.label_file[0])
        label = f[f['ID'] == filename]['DNA'].values

        Img = LoadImg1(os.path.join(root, 'image', f'{filename}.nii.gz'))
        t1ce = Img[:, :, :, 0]
        t1ce = norm(t1ce)
        t2 = Img[:, :, :, 1]
        t2 = norm(t2)
        mask = LoadImg1(os.path.join(root, 'OMT-Tumor', f'{filename}.nii.gz'))
        # mask = norm(mask)
        # mask[mask > 0] = 1
        # mask = labelchange(mask, 4, 10)
        image = np.array([t1ce, t2, mask])

        fe = pd.read_csv(self.opt.feature_file[0])
        # print(filename)
        feature1 = fe[fe['ID'] == filename].values[0][1:]
        # print(feature1)
        feature1 = feature1.astype(float).reshape(-1, 1)
        zscore = preprocessing.StandardScaler()
        # 标准化处理
        feature1 = zscore.fit_transform(feature1)
        feature1 = feature1.flatten()

        ft = pd.read_csv(self.opt.tensor_feature_file[0])
        feature2 = ft[ft['ID'] == filename].values[0][2:]
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


class omtMGMTDataset(Dataset):
    def __init__(self, file_ls, idx_list, opt, transform=None):
        super(omtMGMTDataset, self).__init__()
        self.file_list = file_ls
        self.idx_list = idx_list
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        # omt_file = random.choice(self.data_file)
        # ID = int(self.idx_list[i])
        # filename = f'{self.opt.dataset}_{ID:0>5}'

        filename = self.idx_list[i]
        root = self.file_list
        f = pd.read_csv(self.opt.label_file[0])
        label = f[f['ID'] == filename]['MGMT'].values

        Img = LoadImg1(os.path.join(root, 'image', f'{filename}.nii.gz'))
        t1ce = Img[:, :, :, 0]
        t1ce = norm(t1ce)
        t2 = Img[:, :, :, 1]
        t2 = norm(t2)
        mask = LoadImg1(os.path.join(root, 'OMT-Tumor', f'{filename}.nii.gz'))
        image = np.array([t1ce, t2, mask])

        # fe = pd.read_csv(self.opt.feature_file[0])
        # # print(filename)
        # feature1 = fe[fe['ID'] == filename].values[0][1:]
        # # print(feature1)
        # feature1 = feature1.astype(float).reshape(-1, 1)
        # zscore = preprocessing.StandardScaler()
        # # 标准化处理
        # feature1 = zscore.fit_transform(feature1)
        # feature1 = feature1.flatten()
        #
        # ft = pd.read_csv(self.opt.tensor_feature_file[0])
        # feature2 = ft[ft['ID'] == filename].values[0][2:]
        # feature2 = feature2.astype(float).reshape(-1, 1)
        # feature2 = np.array(feature2.flatten())

        data = {
            'ID': filename,
            'image': image,
            'label': label,
            # 'feature1': feature1,
            # 'feature2': feature2
        }

        if self.transform is not None:
            data = self.transform(data)
        # data = {datas.keys[i],datas.values[i]}
        # print(len(data))
        return data


class omtGradeDataset(Dataset):
    def __init__(self, file_ls, idx_list, opt, transform=None):
        super(omtGradeDataset, self).__init__()
        self.file_list = file_ls
        self.idx_list = idx_list
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        # omt_file = random.choice(self.data_file)
        # ID = int(self.idx_list[i])
        # filename = f'{self.opt.dataset}_{ID:0>5}'

        filename = self.idx_list[i]
        root = self.file_list
        f = pd.read_csv(self.opt.label_file[0])
        grade = f[f['ID'] == filename]['Grade'].values
        label = HLGG(grade)

        Img = LoadImg1(os.path.join(root, 'image', f'{filename}.nii.gz'))
        t1ce = Img[:, :, :, 0]
        t1ce = norm(t1ce)
        t2 = Img[:, :, :, 1]
        t2 = norm(t2)
        mask = LoadImg1(os.path.join(root, 'OMT-Tumor', f'{filename}.nii.gz'))
        image = np.array([t1ce, t2, mask])

        fe = pd.read_csv(self.opt.feature_file[0])
        # print(filename)
        feature1 = fe[fe['ID'] == filename].values[0][1:]
        # print(feature1)
        feature1 = feature1.astype(float).reshape(-1, 1)
        zscore = preprocessing.StandardScaler()
        # 标准化处理
        feature1 = zscore.fit_transform(feature1)
        feature1 = feature1.flatten()

        ft = pd.read_csv(self.opt.tensor_feature_file[0])
        feature2 = ft[ft['ID'] == filename].values[0][2:]
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
def get_train_val_test_dataset(train_id, val_id, test_id, data_file, opt):
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

    train_set = UCSFDataset(data_file[0], train_id, opt, train_transform)
    val_set = UCSFDataset(data_file[0], val_id, opt, val_transforms)
    test_set = UCSFDataset(data_file[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set


def get_train_val_test_omt_dataset(train_id, val_id, test_id, data_file, opt):
    train_transform = Compose([
        ToTensord(keys=['image', 'feature1', 'feature2'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64),
        # RandRot90(keys=['image'], p=opt.prob_rot90),
        # RandFliplr(keys=['image'], p=opt.prob_fliplr),
        # RandFlipud(keys=['image'], p=opt.prob_flipud),
        # RandomGaussianNoise(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        ToTensord(keys=['image', 'feature1', 'feature2'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64)
    ])

    train_set = UCSFomtDataset(data_file[0], train_id, opt, train_transform)
    val_set = UCSFomtDataset(data_file[0], val_id, opt, val_transforms)
    test_set = UCSFomtDataset(data_file[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set


# def get_val_dataset(opt):
#     val_id = list(pd.read_csv(os.path.join('index', f'fold_{opt.fold}.csv'))[opt.index].dropna())
#     val_transforms = Compose([
#         LoadImg(keys=['image', 'label', 'rawlab']),
#         LoadRawImg(keys=['rawimg']),
#         LoadIdx(keys=['Idx', 'InvIdx']),
#         ImgNormal(keys=['image']),
#         ConvertToTumor(keys=['label', 'rawlab'], tumor=opt.tumor),
#         ToTensord(keys=['image', 'label', 'rawimg', 'rawlab', 'Idx', 'InvIdx'], dtype=torch.float),
#         ChannelFirst(keys=['image']),
#         GetNonzero(keys=['rawimg'])
#     ])
#     val_set = BraTSDataset(opt.val_file, val_id, opt, val_transforms)
#
#     return val_set
#
#
# def get_inference_dataset(opt):
#     inference_id = list(pd.read_csv(os.path.join('index', 'fold_0.csv'))[opt.index].dropna())
#     inference_transforms = Compose([
#         LoadImg(keys=['image']),
#         LoadRawImg(keys=['rawimg']),
#         LoadIdx(keys=['Idx', 'InvIdx']),
#         ImgNormal(keys=['image']),
#         ToTensord(keys=['image', 'rawimg', 'Idx', 'InvIdx'], dtype=torch.float),
#         ChannelFirst(keys=['image']),
#         GetNonzero(keys=['rawimg'])
#     ])
#     infernece_set = BraTSDataset(opt.inference_file, inference_id, opt, inference_transforms)
#
#     return infernece_set


def get_train_val_test_omt_1p19qdataset(train_id, val_id, test_id, file_ls, data_loc, opt):
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

    train_set = omt1p19qDataset(file_ls[0], train_id, opt, train_transform)
    val_set = omt1p19qDataset(data_loc[0], val_id, opt, val_transforms)
    test_set = omt1p19qDataset(data_loc[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set


def get_train_val_test_omt_MGMTdataset(train_id, val_id, test_id, file_ls, data_loc, opt):
    train_transform = Compose([
        ToTensord(keys=['image'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64),
        RandRot90(keys=['image'], p=opt.prob_rot90),
        RandFliplr(keys=['image'], p=opt.prob_fliplr),
        RandFlipud(keys=['image'], p=opt.prob_flipud),
        RandomGaussianNoise(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        ToTensord(keys=['image'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64)
    ])

    train_set = omtMGMTDataset(file_ls[0], train_id, opt, train_transform)
    val_set = omtMGMTDataset(data_loc[0], val_id, opt, val_transforms)
    test_set = omtMGMTDataset(data_loc[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set


def get_train_val_test_omt_Gradedataset(train_id, val_id, test_id, file_ls, data_loc, opt):
    train_transform = Compose([
        ToTensord(keys=['image'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64),
        RandRot90(keys=['image'], p=opt.prob_rot90),
        RandFliplr(keys=['image'], p=opt.prob_fliplr),
        RandFlipud(keys=['image'], p=opt.prob_flipud),
        RandomGaussianNoise(keys=['image'], p=opt.prob_noise)
    ])

    val_transforms = Compose([
        ToTensord(keys=['image'], dtype=torch.float),
        ToTensord(keys=['label'], dtype=torch.int64)
    ])

    train_set = omtGradeDataset(file_ls[0], train_id, opt, train_transform)
    val_set = omtGradeDataset(data_loc[0], val_id, opt, val_transforms)
    test_set = omtGradeDataset(data_loc[0], test_id, opt, val_transforms)
    return train_set, val_set, test_set


# ============================== Crop raw data ==============================
# def get_train_val_crop_dataset(opt):
#     train_id = list(pd.read_csv(os.path.join('index', f'fold_{opt.fold}.csv')).train_id)[:opt.train_num]
#     val_id = list(pd.read_csv(os.path.join('index', f'fold_{opt.fold}.csv')).val_id)[:opt.val_num]
#
#     train_transforms = Compose([
#         LoadImg(keys=['rawlab']),
#         LoadRawImg(keys=['rawimg']),
#         ImgNormal(keys=['rawimg']),
#         ConvertToTumor(keys=['rawlab'], tumor=opt.tumor),
#         ToTensord(keys=['rawimg', 'rawlab'], dtype=torch.float),
#         ChannelFirst(keys=['rawimg']),
#         CropForegroundd(keys=['rawimg', 'rawlab'], source_key='rawimg'),
#
#         RandSpatialCropSamplesd(
#             keys=['rawimg', 'rawlab'],
#             roi_size=(opt.size, opt.size, opt.size),
#             num_samples=2,
#             random_size=False
#         ),
#
#         # RandCropByPosNegLabeld(
#         #    keys=['rawimg', 'rawlab'],
#         #    label_key='rawlab',
#         #    spatial_size=(opt.size, opt.size, opt.size),
#         #    pos=1, neg=1,
#         #    num_samples=2,
#         #    image_key='rawimg',
#         #    image_threshold=0),
#
#         RandRot90(keys=['rawimg', 'rawlab'], p=opt.prob_rot90),
#         RandFliplr(keys=['rawimg', 'rawlab'], p=opt.prob_fliplr),
#         RandFlipud(keys=['rawimg', 'rawlab'], p=opt.prob_flipud),
#         RandomGaussianNoise(keys=['rawimg'], p=opt.prob_noise),
#         EnsureTyped(keys=['rawimg', 'rawlab'])
#     ])
#
#     val_transforms = Compose([
#         LoadImg(keys=['rawlab']),
#         LoadRawImg(keys=['rawimg']),
#         ImgNormal(keys=['rawimg']),
#         ConvertToTumor(keys=['rawlab'], tumor=opt.tumor),
#         ToTensord(keys=['rawimg', 'rawlab'], dtype=torch.float),
#         ChannelFirst(keys=['rawimg']),
#         CropForegroundd(keys=['rawimg', 'rawlab'], source_key='rawimg'),
#         EnsureTyped(keys=['rawimg', 'rawlab'])
#     ])
#
#     train_set = BraTSDataset(opt.train_file, train_id, opt, train_transforms)
#     val_set = BraTSDataset(opt.val_file, val_id, opt, val_transforms)
#
#     return train_set, val_set
#
#
# def get_inference_crop_dataset(opt):
#     inference_id = list(pd.read_csv(os.path.join('index', 'fold_0.csv'))[opt.index].dropna())
#     inference_transforms = Compose([
#         LoadRawImg(keys=['rawimg']),
#         ImgNormal(keys=['rawimg']),
#         ToTensord(keys=['rawimg'], dtype=torch.float),
#         ChannelFirst(keys=['rawimg']),
#         CropForegroundd(keys=['rawimg'], source_key='rawimg'),
#         EnsureTyped(keys=['rawimg'])
#     ])
#
#     inference_set = BraTSDataset(opt.inference_file, inference_id, opt, inference_transforms)
#
#     return inference_set

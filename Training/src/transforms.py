import os
import torch
import random
import numpy as np
import nibabel as nib
import scipy.io as sio



class LoadImg():
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                nifti = nib.load(data[key])
                data[key] = nifti.get_fdata()
            
            else:
                raise KeyError(f"{key} is not a key of data")
                
        return data


class LoadIdx():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = sio.loadmat(data[key])[key]

            else:
                raise KeyError(f"{key} is not a key of data")

        return data


class GetNonzero():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = data[key].mean(dim=-1) > 0

            else:
                raise KeyError(f"{key} is not a key of data")

        return data
        

class ImgNormal():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                mean = np.mean(data[key], axis=(0,1,2))
                std = np.std(data[key], axis=(0,1,2))
                z = (data[key]-mean) / std
                rmax, rmin = 5, -5
                z[z>rmax] = rmax
                z[z<rmin] = rmin
                data[key] = (z-rmin) / (rmax-rmin)

            else:
                raise KeyError(f"{key} is not a key of data")

        return data


class ConvertToTumor():
    def __init__(self, keys, tumor):
        self.keys = keys
        self.tumor = tumor
        self.label = {'WT':[1,2,4], 'TC':[1,4], 'ET':[4]}

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if self.tumor == 'all':
                    wt = np.isin(data[key], self.label['WT']).astype(np.int16)
                    tc = np.isin(data[key], self.label['TC']).astype(np.int16)
                    et = np.isin(data[key], self.label['ET']).astype(np.int16)

                    data[key] = np.stack([wt, tc, et], axis=0)

                else:  # WT, TC, ET
                    tmp = np.isin(data[key], self.label[self.tumor])
                    data[key] = np.expand_dims(tmp.astype(np.int16), axis=0) ## add channel

            else:
                raise KeyError(f"{key} is not a key of data")

        return data


class ChannelFirst():
    def __init__(self, keys):
        self.keys = keys
        
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = data[key].permute(3, 0, 1, 2)

            else:
                raise KeyError(f"{key} is not a key of data")
                
        return data


class RandRot90():
    def __init__(self, keys, p=0.5):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    data[key] = torch.rot90(data[key], k=1, dims=(1,2))

                else:
                    raise KeyError(f"{key} is not a key of data")
                
        return data


class RandFliplr():
    def __init__(self, keys, p=0.5):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    tmp = data[key]
                    fliplr_tensor = torch.zeros_like(tmp)
                    for i in range(tmp.shape[0]):
                        fliplr_tensor[i, :, :, :] = torch.fliplr(tmp[i, :, :, :])

                    data[key] = fliplr_tensor

                else:
                    raise KeyError(f"{key} is not a key of data")
                
        return data


class RandFlipud():
    def __init__(self, keys, p=0.5):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    tmp = data[key]
                    flipud_tensor = torch.zeros_like(tmp)
                    for i in range(tmp.shape[0]):
                        flipud_tensor[i, :, :, :] = torch.flipud(tmp[i, :, :, :])
                    
                    data[key] = flipud_tensor

                else:
                    raise KeyError(f"{key} is not a key of data")

        return data


class RandomGaussianNoise():
    def __init__(self, keys, p=0.1, sig=0.01):
        self.keys = keys
        self.sig = sig
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            for key in self.keys:
                if key in data:
                    data[key] += self.sig * torch.randn(data[key].shape)

                else:
                    raise KeyError(f"{key} is not a key of data")
        return data


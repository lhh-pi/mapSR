import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


# from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        # hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #
        # if self.sample_q is not None:
        #     sample_lst = np.random.choice(
        #         len(hr_coord), self.sample_q, replace=False)
        #     hr_coord = hr_coord[sample_lst]
        #     hr_rgb = hr_rgb[sample_lst]
        #
        # cell = torch.ones_like(hr_coord)
        # cell[:, 0] *= 2 / crop_hr.shape[-2]
        # cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            # 'coord': hr_coord,
            # 'cell': cell,
            # 'gt': hr_rgb
            'gt': crop_hr
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')  # 下采样
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        # self.scale_min = scale_min
        # self.scale_min = scale_max
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # s = random.uniform(self.scale_min, self.scale_max)
        s = self.scale_max  # 下采样倍率

        if self.inp_size is None:  # N*H*W*C
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]  # 原始图像GT
            img_down = resize_fn(img, (h_lr, w_lr))  # 下采样图像LR
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size  # 裁剪出patch_size = inp_size的小块
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)  # H
            y0 = random.randint(0, img.shape[-1] - w_hr)  # W
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]  # 裁剪出原始图像GT
            crop_lr = resize_fn(crop_hr, w_lr)  # 对应区域LR

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        # hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #
        # if self.sample_q is not None:
        #     sample_lst = np.random.choice(
        #         len(hr_coord), self.sample_q, replace=False)
        #     hr_coord = hr_coord[sample_lst]
        #     hr_rgb = hr_rgb[sample_lst]

        # cell = torch.ones_like(hr_coord)
        # cell[:, 0] *= 2 / crop_hr.shape[-2]
        # cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            # 'coord': hr_coord,
            # 'cell': cell,
            # 'gt': hr_rgb
            'gt': crop_hr
        }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:03:35 2019

@author: siddhesh
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import torch
from albumentations import (
    RandomBrightnessContrast,
    HueSaturationValue,
    RandomGamma,
    GaussNoise,
    GaussianBlur,
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Normalize,
)
import time
from openslide import OpenSlide
from mpl_toolkits.mplot3d import Axes3D


class GenClassDataset(Dataset):
    def __init__(self, csv_file, params, valid=False):
        self.df = pd.read_csv(csv_file, header=0)
        self.params = params
        self.valid = valid
        self.train_transforms = Compose(
            [
                RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=45, val_shift_limit=30
                ),
                RandomGamma(gamma_limit=(80, 120)),
                GaussNoise(var_limit=(10, 200)),
                GaussianBlur(blur_limit=11),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True, p=1.0
                ),
            ]
        )
        self.validation_transforms = Compose(
            [
                Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True, p=1.0
                )
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, patient_id):
        image_path = os.path.join(self.df.iloc[patient_id, 0])
        image = OpenSlide(image_path)
        x = self.df.iloc[patient_id, 1]
        y = self.df.iloc[patient_id, 2]
        patch = np.array(image.read_region((x, y), 0, (1024, 1024)).convert("RGB"))
        label = self.df.iloc[patient_id, 3]
        del image
        if self.valid:
            image = self.train_transforms(image=patch)
        else:
            image = self.validation_transforms(image=patch)
        patch = image["image"]
        patch = np.transpose(patch, (2, 0, 1))
        patch = torch.FloatTensor(patch)
        return patch, label


train_csv = ""
params = {}


# fig, ax = plt.subplots(1, figsize=(10, 10))


sub_dummy = []
j = 12
print("#" * 80)
print("Num Workers : ", j)
tstart = time.time()
x = [i for i in range(64, 10241, 64)]
y = []
i = 0
dataset_train = GenClassDataset(train_csv, params, valid=False)
train_loader = DataLoader(
    dataset_train, batch_size=32, shuffle=True, num_workers=j, pin_memory=False
)
start = time.time()
print(len(train_loader))
for batch_idx, (image_data, label) in enumerate(train_loader):
    del image_data, label
    if i >= len(x):
        print("i went too far, It had to be done.")
        break
    if batch_idx * 32 == x[i]:
        y.append(time.time() - start)
        print("Number of Images : ", x[i], i)
        print("Time Taken : ", (time.time() - start))
        i += 1


tend = time.time()
print("Time taken for {} workers : {}".format(j, tend - start))
y = np.array(y)
x = np.array(x)
np.savez_compressed("./Cluster_300", x=x, y=y)

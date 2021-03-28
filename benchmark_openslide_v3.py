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
from tqdm import tqdm


class GenClassDataset(Dataset):
    def __init__(self, csv_file, ref_file, params, valid=False):
        self.csv_file = csv_file
        self.ref_file = ref_file
        self.df = pd.read_csv(csv_file)
        self.params = params
        self.valid = valid
        self.openslide_obs = {}
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
        self.reset_slideobjects()

    def reset_slideobjects(self):
        print("Resetting")
        temp_df = pd.read_csv(self.ref_file)
        for i in tqdm(range(temp_df.shape[0])):
            pid = temp_df.iloc[i, 0]
            path = temp_df.iloc[i, 1]
            self.openslide_obs[pid] = OpenSlide(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, patient_id):
        pid = self.df.loc[patient_id, "PID"]
        x = int(self.df.loc[patient_id, "x_loc"])
        y = int(self.df.loc[patient_id, "y_loc"])
        slide_ob = self.openslide_obs[pid]
        patch = np.array(slide_ob.read_region((x, y), 0, (1024, 1024)).convert("RGB"))
        label = self.df.loc[patient_id, "label"]
        if self.valid:
            image = self.train_transforms(image=patch)
        else:
            image = self.validation_transforms(image=patch)
        patch = image["image"]
        patch = np.transpose(patch, (2, 0, 1))
        patch = torch.FloatTensor(patch)
        return patch, label


train_csv = "/cbica/home/thakurs/comp_space/projects/Histo-Seg/Colon-Data/Hipagen/CSV/Train_fold_1.csv"
ref_csv = "/cbica/home/thakurs/comp_space/projects/Histo-Seg/Colon-Data/Hipagen/CSV/reference.csv"
params = {}

sub_dummy = []
j = 12
print("#" * 80)
print("Num Workers : ", j)
tstart = time.time()
x = [i for i in range(64, 10241, 64)]
y = []
i = 0
dataset_train = GenClassDataset(train_csv, ref_csv, params, valid=False)
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
    if batch_idx == 262144:
        dataset_train.reset_slide_objects()

tend = time.time()
print("Time taken for {} workers : {}".format(j, tend - start))
y = np.array(y)
x = np.array(x)
os.makedirs("./New_Benchmarks_v3", exist_ok=True)
np.savez_compressed("./New_Benchmarks_v3/Cluster_login_node_not_ram", x=x, y=y)

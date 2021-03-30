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
import glob
import argparse
from timm import create_model


def train_model(dataloader, threads):
    start = time.time()
    print(len(train_loader))
    model = create_model("resnet50")
    model.eval()
    print("Currently evaluating : {} threads".format(threads))
    for batch_idx, (image_data, label) in enumerate(train_loader):
        output = model(image_data)
        if batch_idx >= 300:
            break

    tend = time.time()
    print("Time taken for {} workers : {}".format(threads, tend - start))

    time_taken = tend - tstart

    return time_taken


class GenClassDataset(Dataset):
    def __init__(self, csv_file, ref_file, params, valid=False):
        self.csv_file = csv_file
        self.ref_file = ref_file
        self.df = pd.read_csv(csv_file)
        self.ref_df = pd.read_csv(ref_file)
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
        pid = self.df.loc[patient_id, "PID"]
        x = int(self.df.loc[patient_id, "x_loc"])
        y = int(self.df.loc[patient_id, "y_loc"])
        image_path = self.ref_df[self.ref_df["PID"] == pid]["Image_Path"].values[0]
        slide_ob = OpenSlide(image_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="input path of tissue files",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--ref_path",
        dest="ref_path",
        help="reference path of output files",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help="Number of threads to test against",
        required=True,
    )

    args = parser.parse_args()

    train_csv = os.path.abspath(args.input_path)
    ref_csv = os.path.abspath(args.ref_path)
    params = {}

    sub_dummy = []
    threads = int(args.threads)
    print("#" * 80)
    print("Num Workers : ", threads)
    tstart = time.time()

    total_threads = []
    thread_time_taken = []

    for thread in range(threads):

        dataset_train = GenClassDataset(train_csv, ref_csv, params, valid=False)
        train_loader = DataLoader(
            dataset_train,
            batch_size=32,
            shuffle=True,
            num_workers=thread,
            pin_memory=False,
        )

        time_taken = train_model(train_loader, threads)
        total_threads.append(thread)
        thread_time_taken.append(thread)

    os.makedirs("./openslide_v1", exist_ok=True)
    np.savez_compressed(
        "./New_Benchmarks_v3/gpu_stress_test_v1_" + str(args.threads) + "_threads",
        thread_list=total_threads,
        time_taken=thread_time_taken,
    )

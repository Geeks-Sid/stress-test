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
import argparse
import os
import torch
from multiprocessing import Pool, cpu_count, Manager
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
import pyvips
from tqdm import tqdm
import concurrent.futures
import time
import threading
from torchvision.models import resnet50

pyvips.cache_set_max(0)


class GenClassDataset(Dataset):
    def __init__(self, csv_file, ref_file, params, valid=False):
        self.csv_file = csv_file
        self.ref_file = ref_file
        self.df = pd.read_csv(csv_file)
        self.patients = pd.read_csv(ref_file)
        self.params = params
        self.n_processes = params["threads"]
        self.valid = valid

        self.pyvips_objs = {}
        self._lock = threading.Lock()

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
        print("Initializing Dataset")
        self.intialize_slideobjects()
        print("Done Initializing")

    def intialize_slideobjects(self):
        print("Resetting")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.params["threads"]
        ) as executor:
            for index in range(self.params["threads"]):
                executor.submit(self.load_slides, index)

    def load_slides(self, k):
        print("In here Loading Slide thread no. : ", k)
        sub_patients = pd.DataFrame(columns=self.patients.columns)
        for i in range(len(self.patients)):
            if i % self.n_processes == k:
                sub_patients = sub_patients.append(
                    self.patients.iloc[i, :], ignore_index=True
                )
        slides_to_add = {}
        for i in range(sub_patients.shape[0]):
            # This can definitely be optimized
            pid = sub_patients.iloc[i, 0]
            path = sub_patients.iloc[i, 1]
            # Add a value to the dictionary with the PID as the key and the path as the value
            slides_to_add[pid] = path
        # Go through this data partition, actually create the objects, then update main dictionary after objects are created.
        self.add_pyvips_objects(slides_to_add)

    # Opens slides, passes to update method
    def add_pyvips_objects(self, pid_path_dict):
        new_dict = {}
        # Iterate over the pids and paths, and actually create the pyvips objects.
        for pid, path in pid_path_dict.items():
            # Create the object, insert into new dictionary
            # Update this to be a parameter for which the level is to be fetched
            new_dict[pid] = pyvips.Image.openslideload(path, level=0)
            print("Updating lock...")
        # After new dictionary is created, push to main dictionary
        self.locked_update(new_dict)

    # Need to install threadlock to prevent race conditioning
    # This is what actually adds them to the shared dictionary.
    def locked_update(self, new_dict):
        with self._lock:
            print("Locking and updating...")
            # Create copy, update dictionary with new values (pids => pyvips objects)
            local_copy = self.pyvips_objs
            local_copy.update(new_dict)
            # After update, restore and release lock
            print("Main dictionary updated. Lock released.")
            self.pyvips_objs = local_copy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, patient_id):
        pid = self.df.iloc[patient_id, 0]
        x = int(self.df.iloc[patient_id, 1])
        y = int(self.df.iloc[patient_id, 2])
        slide_ob = self.pyvips_objs[pid]
        patch = np.array(slide_ob.fetch((x, y), 0, (1024, 1024)).convert("RGB"))
        label = self.df.iloc[patient_id, 3]
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
        help="input path for the csv",
        required=True,
    )
    parser.add_argument(
        "-r", "--ref_path", dest="ref_path", help="path for the reference csv"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path for the landmark files",
        required=True,
    )
    parser.add_argument(
        "-t", "--threads", dest="threads", help="number of threads, by default will use"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    ref_path = os.path.abspath(args.ref_path)
    output_path = os.path.abspath(args.output_path)
    sub_dummy = []
    j = int(args.threads)
    print("#" * 80)
    print("Num Workers : ", j)
    tstart = time.time()
    x = [i for i in range(64, 131073, 64)]
    y = []
    i = 0
    params = {}
    params["threads"] = j
    dataset_train = GenClassDataset(input_path, ref_path, params, valid=False)
    train_loader = DataLoader(
        dataset_train, batch_size=32, shuffle=True, num_workers=j, pin_memory=False
    )
    start = time.time()
    model = resnet50(num_class=2, imagenet=True)
    print(len(train_loader))
    for batch_idx, (image_data, label) in enumerate(train_loader):
        output = model(image_data)
        loss = nn.CrossEntropyLoss(output, label)
        if i >= len(x):
            print("i went too far, It had to be done.")
            break
        if batch_idx * 32 == x[i]:
            y.append(time.time() - start)
            print("Number of Images : ", x[i], i)
            print("Time Taken : ", (time.time() - start))
            i += 1
        if batch_idx == 65536:
            dataset_train.reset_slide_objects()
    tend = time.time()
    print("Time taken for {} workers : {}".format(j, tend - start))
    y = np.array(y)
    x = np.array(x)
    os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_path, "Cluster_login_node_not_ram_multiloader", x=x, y=y)
    )

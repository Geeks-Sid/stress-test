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
import time
# from openslide import OpenSlide
import argparse
# from timm import create_model
from torch.cuda.amp import autocast

from GANDLF.utils import *
from GANDLF.parseConfig import *
from GANDLF.parameterParsing import *
from GANDLF.data.ImagesFromDataFrame import ImagesFromDataFrame


def train_model(dataloader, thread):
    start = time.time()
    print(len(train_loader))
    # model = create_model("resnet18")
    model = get_model('resunet', 3, 4, 4, 30, 'softmax', psize = [128,128,128], batch_size = 1)
    model 
    model.eval()
    print("Currently evaluating : {} thread".format(thread))
    print("Length of Train Loader : ", len(train_loader))
    model.cuda()
    for batch_idx, (image_data, label) in enumerate(train_loader):
        image_data = image_data.cuda()
        with autocast():
            output = model(image_data)
        if batch_idx >= 50:
            break
        del image_data, output
    tend = time.time()
    print("Time taken for {} workers : {}".format(thread, tend - start))

    time_taken = tend - start

    return time_taken

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
        "-t",
        "--threads",
        dest="threads",
        help="Number of threads to test against",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        help="Standard batch size to be for dataloader",
    )

    args = parser.parse_args()

    train_csv = os.path.abspath(args.input_path)
    ref_csv = os.path.abspath(args.ref_path)
    threads = int(args.threads)
    batch_size = int(args.batch_size)
    patch_size = int(args.patch_size)

    params = parseConfig('./GANDLF/samples/config_segmentation_brats.yaml')
    params["batch_size"] = batch_size

    print("#" * 80)
    print("Num Workers : ", threads)
    tstart = time.time()

    total_threads = []
    thread_time_taken = []

    data_train, headers_train = parseTrainingCSV(train_csv)

    for thread in range(threads, 4, -1):
        try:
            dataset_train = ImagesFromDataFrame(data_train, 
                        params["patch_size"], 
                        headers_train, 
                        q_max_length = 100, 
                        q_samples_per_volume = 5, 
                        q_num_workers = thread, 
                        q_verbose = False, 
                        sampler = 'uniform', 
                        train = True, 
                        augmentations = params["data_augmentations"], 
                        preprocessing = params["data_preprocessing"], 
                        in_memory = False)

            train_loader = DataLoader(
                dataset_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=thread,
                pin_memory=False,
            )

            time_taken = train_model(train_loader, thread)
            total_threads.append(thread)
            thread_time_taken.append(time_taken)
        except RuntimeError:
            total_threads.append(thread)
            thread_time_taken.append(0)
        torch.cuda.empty_cache()

    os.makedirs("./openslide_v1", exist_ok=True)
    np.savez_compressed(
        "./openslide_v1/gpu_stress_test_v1_"
        + str(threads)
        + "_threads_"
        + str(batch_size)
        + "_batch_size_"
        + str(patch_size)
        + "_patch_size",
        thread_list=total_threads,
        time_taken=thread_time_taken,
        batch_size=batch_size,
        patch_size=patch_size,
    )

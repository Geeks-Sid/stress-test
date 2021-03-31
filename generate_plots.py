#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 03:12:08 2021

@author: siddhesh
"""

import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = [6, 24, 96, 384]
patch_size = [4096, 2048, 1024, 512]

thread_array = []
time_array = []

for batch, patch in zip(batch_size, patch_size):
    x = np.load(
        "./openslide_v1/gpu_stress_test_v1_28_threads_"
        + str(batch)
        + "_batch_size_"
        + str(patch)
        + "_patch_size.npz"
    )
    thread_array.append(x['thread_list'])
    time_array.append(x['time_taken'])

thread_array = np.array(thread_array)
time_array = np.array(time_array)
temp_array = np.zeros_like(time_array)

for i in range(len(batch_size)):
    for j in range(time_array.shape[1]-1, 0, -1):
        # print(str(time_array[i, j]), '=', time_array[i, j], '-', time_array[i, j-1])
        temp_array[i, j] = time_array[i, j] - time_array[i, j-1]

temp_array[:, 0] = time_array[:, 0]

for i in range(len(batch_size)):
    plt.plot(thread_array[i, ...], temp_array[i, ...])

plt.ylim(0)
legend_strings = [str(i)+'x'+str(j) for i,j in zip(batch_size, patch_size)]
plt.legend(legend_strings)
plt.show()

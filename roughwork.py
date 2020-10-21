#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:20 2020

@author: jakeyap
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import DataProcessor

MAX_POST_PER_THREAD = 4
MAX_POST_LENGTH = 256
directory = './data/combined/'
train_filename = 'encoded_shuffled_train_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
train_dataframe = DataProcessor.load_from_pkl(directory + train_filename)

temp0 = train_dataframe.labels_list
lengths = []
for each_list in temp0:
    lengths.append(len(each_list))

fig,axes = plt.subplots(2,1,sharex=True)
ax0 = axes[0]
ax1 = axes[1]

lengths = torch.tensor(lengths)
median1 = torch.median(lengths)
median1 = median1.item()
lo_side1 = min(lengths)
hi_side1 = max(lengths)
bins1 = np.arange(lo_side1, hi_side1)

temp2 = torch.tensor(train_dataframe.orig_length)
median2 = torch.median(temp2)
median2 = median2.item()
lo_side2 = min(temp2)
hi_side2 = max(temp2)
bins2 = np.arange(lo_side2, hi_side2)
temp2 = temp2.numpy()

ax0.hist(x=temp2, bins=bins2, width=0.8)
ax0.grid(True)
ax0.set_title('Raw Conversation lengths in training set. Median is %d' % median2)
ax0.set_yscale('log')

lengths = lengths.numpy()
ax1.hist(x=lengths, bins=bins1, width=0.8)
plt.title('Conversation lengths after prune. Median is %d' % median1)
ax1.grid(True)
ax1.set_yscale('log')

plt.tight_layout()
print('hello world')
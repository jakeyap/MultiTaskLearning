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


# handling the filtered dataset
temp0 = train_dataframe.labels_list
len_filt = []

for each_list in temp0:
    len_filt.append(len(each_list))

fig,axes = plt.subplots(2,1,sharex=True)
ax0 = axes[0]
ax1 = axes[1]

len_filt = torch.tensor(len_filt)
median1 = torch.median(len_filt)
median1 = median1.item()

lo_side1 = min(len_filt)
hi_side1 = max(len_filt)
bins1 = hi_side1.item() - lo_side1.item()

# For handling the original dataset

len_orig = torch.tensor(train_dataframe.orig_length)
median2 = torch.median(len_orig)
median2 = median2.item()
lo_side2 = min(len_orig)
hi_side2 = max(len_orig)
bins2 = hi_side2.item() - lo_side2.item()
len_orig = len_orig.numpy()

ax0.hist(x=len_orig, bins=bins2, width=0.8)
ax0.grid(True)
ax0.set_title('Raw Conversation lengths in training set. Median is %d' % median2)
ax0.set_yscale('log')

len_filt = len_filt.numpy()
ax1.hist(x=len_filt, bins=bins1, width=0.8)
plt.title("Convo lengths after pruning ones that don't reply to parent. Median is %d" % median1)
ax1.grid(True)
ax1.set_yscale('log')

plt.tight_layout()

# split into 4 buckets
median_mid = torch.median(torch.tensor(len_orig)).item()
index_lo = len_orig < median_mid
index_hi = ~index_lo

lengths_lo = len_orig[index_lo]
lengths_hi = len_orig[index_hi]

median_lo = torch.median(torch.tensor(lengths_lo)).item()
median_hi = torch.median(torch.tensor(lengths_hi)).item()

ax0.plot([median_mid,median_mid],[1,1000], color='black', lw=1, linestyle='dashed')
ax0.plot([median_lo,median_lo],[1,1000], color='black', lw=1, linestyle='dashed')
ax0.plot([median_hi,median_hi],[1,1000], color='black', lw=1, linestyle='dashed')

posts0 = len_orig < median_lo
posts1 = (len_orig >= median_lo) & (len_orig < median_mid)
posts2 = (len_orig >= median_mid) & (len_orig < median_hi)
posts3 = (len_orig >= median_hi)

count0 = posts0.sum()
count1 = posts1.sum()
count2 = posts2.sum()
count3 = posts3.sum()

print(count0, count1, count2, count3)
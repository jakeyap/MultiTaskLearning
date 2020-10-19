#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:20 2020

@author: jakeyap
"""

import torch
import matplotlib.pyplot as plt
'''
train_filename = 'encoded_shuffled_train_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
train_dataframe = DataProcessor.load_df_from_pkl(directory + train_filename)

temp1 = torch.tensor(train_dataframe.orig_length)
median = torch.median(temp1)

lo_side = min(temp1)
hi_side = max(temp1)
bins = torch.range(lo_side-0.5, hi_side+0.5)
plt.hist(x=temp1, bins=bins, width=0.8)
plt.title('Conversation lengths in training set. Median is %d' % median.item())
plt.grid(True)
plt.tight_layout()
plt.yscale('log')
'''
print('hello world')
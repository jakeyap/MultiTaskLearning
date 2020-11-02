#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:13:10 2020

@author: jakeyap
"""

import torch
from transformers.modeling_bert import BertConfig, BertEncoder

config = BertConfig('bert-base-uncased')
layer = BertEncoder(config)

num_hidden = 768
len_encode = 256
batch_size = 2
torch.cuda.empty_cache()

layer.to('cuda')
layer.eval()

input1 = torch.rand(size=(batch_size, len_encode, num_hidden))
output1= layer(input1.to('cuda'))
output1= output1[0].to('cpu')

input2 = torch.rand(size=(batch_size, len_encode * 4, num_hidden))
output2= layer(input2.to('cuda'))
output2= output2[0].to('cpu')

print(output1.shape)
print(output2.shape)
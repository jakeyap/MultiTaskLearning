#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:20 2020

@author: jakeyap
"""

import torch

temp1 = torch.randint(low=0,high=4,size=(10,))
temp2 = torch.tensor([i.index for i in temp1 if i==0])
temp3 = temp1[temp2]

print(temp1)
print(temp2)
print(temp3)
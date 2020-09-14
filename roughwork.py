#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:20 2020

@author: jakeyap
"""

import torch.nn as nn

class BaseClass(nn.Module):
    def __init__(self, stance_num_labels=4, length_num_labels=2, max_post_num=4, max_post_length=64):
        super(BaseClass, self).__init__()
        self.apply(1)
        
if __name__ == '__main__':
    obj1 = BaseClass()
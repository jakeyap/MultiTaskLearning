#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:20:15 2020
This script converts the coarse discourse json file to the same style as the
semeval17 format, where each thread is a row and '|||||' separates posts. 

Labels are given as a list [] in front of the posts. 
0 = deny
1 = support
2 = query
3 = comment
@author: jakeyap
"""

import json
import numpy as np

FILEDIR  = './../data/coarse_discourse/'
FILENAME = 'coarse_discourse_dump_reddit.json'

entries = []
threadlengths = []

file = open(FILEDIR+FILENAME, 'r')
counter = 0
for eachline in file:
    #print(json.dumps(file.readline(), indent=4))
    entries.append(json.loads(eachline))
    counter = counter + 1
file.close()

for eachthread in entries:
    threadlength = len(eachthread['posts'])
    threadlengths.append(threadlength)

import matplotlib.pyplot as plt
plt.hist(threadlengths)
plt.grid(True)
plt.ylabel('Count')
plt.xlabel('Length')
plt.tight_layout()

# figure out the median point of the data lengths
median = np.median(threadlengths)
print(median)
# count how many are below or equal to 9
# count how many are above 9
smallcounter = 0
for length in threadlengths:
    if length <= median:
        smallcounter = smallcounter + 1
largecounter = len(threadlengths) - smallcounter

# TODO: Merge the data
# TODO: Convert posts with no majority label into COMMENT
# TODO: Convert labels
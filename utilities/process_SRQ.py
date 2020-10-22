#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:04:11 2020

@author: jakeyap
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from preprocessor_functions import remove_urls, remove_spaces, post_isempty
from convert_coarse_discourse import generate_sdqc_mapping

import os
import sys
import argparse
import json

def convert_2_semeval_format(filename='./data/srq/stance_dataset.json'):
    print('processing SRQ dataset')
    
    assert os.path.isfile(filename), "SRQ data not found at %s" % filename
    
    file = open(filename,'r')
    lines = file.readlines()
    
    tsv_filename = filename.replace('.json', '_processed.tsv')
    tsvfile = open(tsv_filename, 'w')
    tsvfile.write("index\t#1 Label\t#2 Count\t#3 String\n")
    
    count = 0
    for i in range(len(lines)):
        input_dict = json.loads(lines[i])
        count += 1
        try:
            parent  = clean_posts(input_dict['target_text'])
            reply = clean_posts(input_dict['response_text'])
            
            s1 = parent + ' ||||| ' + reply
            reply_stance = input_dict['label'].lower()
            reply_stance = convert_label_srq_2_sdqc_format(reply_stance)
            root_stance = 1 # root is always labelled comment
            labels = str(root_stance) + ',' + str(reply_stance)
            tsvfile.write("%s\t%s\t%s\t%s\n" % (count, labels, 2, s1))
            if count % 10 == 0:
                print('Done with line %d' % count)
        except KeyError: 
            print('Line %d has missing posts. Skipping.' % count)
    return
    
def convert_label_sdqc_int2str(label_int):
    # ['B-DENY', 'B-SUPPORT', 'B-QUERY', 'B-COMMENT']
    mappinglist = generate_sdqc_mapping()
    return mappinglist[label_int]

def convert_label_sdqc_str2int(label_str):
    mappinglist = generate_sdqc_mapping()
    # ['B-DENY', 'B-SUPPORT', 'B-QUERY', 'B-COMMENT']
    return mappinglist.index(label_str)

def convert_label_srq_2_sdqc_format(labelstring):
    # ['B-DENY', 'B-SUPPORT', 'B-QUERY', 'B-COMMENT']
    labelstring = labelstring.lower()
    if 'denial' in labelstring:
        return 0
    elif 'support' in labelstring:
        return 1
    elif 'queries' in labelstring:
        return 2
    elif 'comment' in labelstring:
        return 3
    else:
        print('Cannot find label')
        raise KeyError

def clean_posts(text):
    if post_isempty(text):
        return '[empty]'
    else:
        text = remove_urls(text)
        text = remove_spaces(text)
        return text

def count_labels():
    filename = "./../data/srq/stance_dataset.json"
    
    file = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = file.readlines()
    labels = []
    
    for i in range(len(lines)):
        input_dict = json.loads(lines[i])
        labelstr = input_dict["label"]
        labelint = convert_label_srq_2_sdqc_format(labelstr)
        thread_labels = [1, labelint]
        labels.extend(thread_labels)
        
    fig, ax = plt.subplots()
    label_list = ['deny','support','query','comment']
    horz = [-0.5, 0.5,1.5,2.5,3.5]
    ax = plt.gca()
    ax.hist(labels,bins=horz, edgecolor='black')
    ax.grid(True)
    ax.set_title('Label Density')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Depth')
    ax.set_xticks([0, 1,2,3])
    ax.set_xticklabels(label_list,size=8, rotation=90)
    fig.tight_layout()
    
    
if __name__ == '__main__':
    filename = './../data/srq/stance_dataset.json'
    convert_2_semeval_format(filename=filename)
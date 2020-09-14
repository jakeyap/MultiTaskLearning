#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:32:49 2020

@author: jakeyap
"""

''' Script for downloading all GLUE data.

Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized, 
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).

mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
'''

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from preprocessor_functions import remove_urls, remove_spaces

import os
import sys
import argparse
import json

def format_rumor(directory):
    print("Processing...SemEval17...stance")

    rumor_train_file = directory + "stance_train.json"
    rumor_dev_file = directory + "stance_dev.json"
    rumor_test_file = directory + "stance_test.json"

    assert os.path.isfile(rumor_train_file), "Train data not found at %s" % rumor_train_file
    assert os.path.isfile(rumor_test_file), "Test data not found at %s" % rumor_test_file

    fin = open(rumor_train_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(directory+"stance_train.tsv", 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 Count\t#3 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = ' ||||| '.join(tweets)
            s1 = remove_urls(s1)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            num_post = len(str_label_lst)
            label = ','.join(str_label_lst)
            train_fh.write("%s\t%s\t%s\t%s\n" % (count, label, num_post, s1))

    fin = open(rumor_dev_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(directory+"stance_dev.tsv", 'w') as train_fh:
        train_fh.write("index\t#1 Label\t#2 Count\t#3 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = ' ||||| '.join(tweets)
            s1 = remove_urls(s1)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            num_post = len(str_label_lst)
            label = ','.join(str_label_lst)
            train_fh.write("%s\t%s\t%s\t%s\n" % (count, label, num_post, s1))

    fin = open(rumor_test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    with open(directory+"stance_test.tsv", 'w') as test_fh:
        test_fh.write("index\t#1 Label\t#2 Count\t#3 String\n")
        count = 0
        for i in range(len(lines)):
            count += 1
            input_dict = json.loads(lines[i])
            tweets = input_dict["tweets"]
            s1 = ' ||||| '.join(tweets)
            s1 = remove_urls(s1)
            label_list = input_dict["label"]
            str_label_lst = [str(p) for p in label_list]
            num_post = len(str_label_lst)
            label = ','.join(str_label_lst)
            test_fh.write("%s\t%s\t%s\t%s\n" % (count, label, num_post, s1))
    print("\tCompleted!")

def plot_thread_depths():
    DATADIR = './../data/semeval17/'
    TEST_FILE   = 'stance_test.json'
    TRAIN_FILE  = 'stance_train.json'
    DEV_FILE    = 'stance_dev.json'
    
    label_lists = [] # list of lists of labels in every thread.
    label_depth = [] # list of ints. each int shows how deep a thread is.
    first_label = [] # list of ints. what the first labels are
    all_labels = [] # list of ints. each int is a label. holds all labels.
    
    for each_filename in [TEST_FILE, TRAIN_FILE, DEV_FILE]:
        filename = DATADIR + each_filename
        jsonfile = open(filename)
        lines = jsonfile.readlines()
        for line in lines:
            thread=json.loads(line)
            label_lists.append(thread['label'])
            label_depth.append(len(thread['label']))
    
    maxdepth = max(label_depth)
    
    median = np.median(label_depth)
    modeitem = scipy.stats.mode(label_depth)
    modeitem = modeitem.mode.item() # deal with scipy formatting
    maxcount = label_depth.count(modeitem)
    
    smallcounter = 0
    for length in label_depth:
        if length <= median:
            smallcounter = smallcounter + 1
    largecounter = len(label_depth) - smallcounter
    
    plt.figure(1)
    plt.hist(label_depth,bins=maxdepth, edgecolor='black')
    plt.grid(True)
    plt.title('Thread depth')
    plt.ylabel('Counts')
    plt.xlabel('Depth')
    plt.tight_layout()
    string=('Median thread length: %2d\n<=median: %d\n>median: %d' 
            % (median, smallcounter, largecounter))
    print(string)
    plt.text(x=maxdepth*0.5,y=maxcount*0.75,s=string,
             bbox=dict(boxstyle="square", fc="white", ec="black", lw=2))
    
    for each_list in label_lists:
        first_label.append(each_list[0])
        all_labels.extend(each_list)
    
    plt.figure(2)
    plt.hist(first_label,bins=4,color='skyblue', lw='0.1', 
             edgecolor='black', rwidth=0.5)
    plt.grid(True)
    plt.title('First post label')
    plt.ylabel('Counts')
    plt.xlabel('Label')
    plt.tight_layout()

if __name__ == '__main__':
    DIRECTORY = './../data/semeval17/'
    format_rumor(DIRECTORY)
    #plot_thread_depths()
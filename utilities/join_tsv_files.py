#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 11 17:32:49 2020

@author: jakeyap
"""
import random

DEST_FOLDER = './../data/combined/'
FOLDER1 = './../data/coarse_discourse/'
FOLDER2 = './../data/semeval17/'

FILENAME1 = 'coarse_discourse_dump_reddit.tsv'
FILENAME2 = 'stance_train.tsv'
FILENAME3 = 'stance_test.tsv'
FILENAME4 = 'stance_dev.tsv'

FILE1 = FOLDER1 + FILENAME1
FILE2 = FOLDER2 + FILENAME2
FILE3 = FOLDER2 + FILENAME3
FILE4 = FOLDER2 + FILENAME4

# aggregate the data
# do own shuffle algorithm
# resplit the files into 80-10-10 ratio
filehandle1 = open(FILE1, 'r')
filehandle2 = open(FILE2, 'r')
filehandle3 = open(FILE3, 'r')
filehandle4 = open(FILE4, 'r')

inputfiles = [filehandle1, filehandle2, filehandle3, filehandle4]

''' create unshuffled data files. merge reddit with original train file '''
trainfile = open(DEST_FOLDER + 'combined_train.tsv', 'w')
testfile = open(DEST_FOLDER + 'combined_test.tsv', 'w')
devfile = open(DEST_FOLDER + 'combined_dev.tsv', 'w')

merged_data = []

# Build the train file
lines = filehandle1.readlines()
header = lines[0]
trainfile.write(header)
# remember to skip header
for eachline in lines[1:]:
    trainfile.write(eachline)
    merged_data.append(eachline)
lines = filehandle2.readlines()
for eachline in lines[1:]:
    trainfile.write(eachline)
    merged_data.append(eachline)

# Build the test file
lines = filehandle3.readlines()
testfile.write(header)
for eachline in lines[1:]:
    testfile.write(eachline)
    merged_data.append(eachline)

# Build the dev file
devfile.write(header)
lines = filehandle4.readlines()
for eachline in lines[1:]:
    devfile.write(eachline)
    merged_data.append(eachline)

''' create shuffled data files. merge all, shuffle, then 80-10-10 '''
shuf_trainfile = open(DEST_FOLDER + 'shuffled_train.tsv', 'w')
shuf_testfile = open(DEST_FOLDER + 'shuffled_test.tsv', 'w')
shuf_devfile = open(DEST_FOLDER + 'shuffled_dev.tsv', 'w')

random.seed(0)
random.shuffle(merged_data)

datalength = len(merged_data)
test_index = int(0.1 * datalength)
dev_index = int(0.2 * datalength)
train_index = int(0.3 * datalength)

test_data = merged_data[0:test_index]
dev_data = merged_data[test_index:dev_index]
train_data = merged_data[dev_index:]

shuf_trainfile.write(header)
shuf_devfile.write(header)
shuf_testfile.write(header)
for eachline in train_data:
    shuf_trainfile.write(eachline)
for eachline in test_data:
    shuf_testfile.write(eachline)
for eachline in dev_data:
    shuf_devfile.write(eachline)
shuf_devfile.close()
shuf_testfile.close()
shuf_trainfile.close()
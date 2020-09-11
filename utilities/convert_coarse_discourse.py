#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:20:15 2020
This script converts the coarse discourse json file to the same style as the
semeval17 format, where each thread is a row and ' ||||| ' separates posts. 

Labels are given as a list [] in front of the posts. 
0 = deny
1 = support
2 = query
3 = comment

mapping table from coarse_discourse into semeval17 format
  Reddit        ==>  NTU
question        ==> query
answer          ==> comment
announcement    ==> comment
agreement       ==> support
appreciation    ==> comment?support
disagreement    ==> deny
-ve reaction    ==> deny? comment?
elaboration     ==> comment
humor           ==> comment
other           ==> comment

@author: jakeyap
"""

import json
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from preprocessor_functions import remove_urls, remove_spaces

def extract_jsonfile(directory, filename):
    entries = []    
    file = open(directory+filename, 'r')
    counter = 0
    for eachline in file:
        line = json.loads(eachline)
        entries.append(line)
        counter = counter + 1
    file.close()
    return entries

def plot_raw_statistics(entries):
    threadlengths = []
    for eachthread in entries:
        threadlength = len(eachthread['posts'])
        threadlengths.append(threadlength)
    
    maxlength = max(threadlengths)
    modeitem = scipy.stats.mode(threadlengths)
    modeitem = modeitem.mode.item() # deal with scipy formatting
    maxcount = threadlengths.count(modeitem)
    plt.figure()
    plt.hist(x=threadlengths, bins=maxlength,edgecolor='black')
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
    string=('Median thread length: %2d\n<=median: %d\n>median: %d' 
            % (median, smallcounter, largecounter))
    print(string)
    plt.text(x=maxlength*0.5,y=maxcount*0.75,s=string,
             bbox=dict(boxstyle="square", fc="white", ec="black", lw=2))
    return threadlengths

def process_first_posts(entries):
    '''
    Process first posts in json file.
        Relabel first posts as SUPPORT class
        Fill in empty fields those with no body in comment
    Returns
    -------
    number of body insertions done (int).

    '''
    errors = 0
    for eachthread in entries:
        posts = eachthread['posts']
        firstpost = posts[0]
        firstpost['majority_type'] = 'B-SUPPORT'
        if 'body' not in firstpost.keys():
            firstpost['body'] = '[empty]'
            errors = errors + 1
        if firstpost['body'] == '':
            firstpost['body'] = '[empty]'
        eachthread['length'] = len(posts)
    return errors
    
def convert_all_labels(entries):
    '''
    Converts all labels into the SDQC format
        Convert posts with no majority label into COMMENT
    Parameters
    ----------
    entries : list of dictionaries
        each list is a reddit thread.
        
    Returns
    -------
    None.

    '''
    for eachthread in entries:
        posts = eachthread['posts']
        for post in posts:
            if 'majority_type' in post.keys():
                label = post['majority_type']
                post['majority_type'] = convert_label_coarse_discourse_2_sdqc(label)
            else:
                # for posts with no annotator agreement, label as other
                post['majority_type'] = 'B-COMMENT'

def convert_label_coarse_discourse_2_sdqc(reddit_label):
    '''
    mapping table from coarse_discourse into semeval17 format
      Reddit        ==>  NTU
    question        ==> query
    answer          ==> comment
    announcement    ==> comment
    agreement       ==> support
    appreciation    ==> comment
    disagreement    ==> deny
    -ve reaction    ==> deny
    elaboration     ==> comment
    humor           ==> comment
    other           ==> comment
    '''
    if reddit_label == 'question': return 'B-QUERY'
    if reddit_label == 'answer': return 'B-COMMENT'
    if reddit_label == 'announcement': return 'B-COMMENT'
    if reddit_label == 'agreement': return 'B-SUPPORT'
    if reddit_label == 'appreciation': return 'B-COMMENT'
    if reddit_label == 'disagreement': return 'B-DENY'
    if reddit_label == 'negativereaction': return 'B-DENY'
    if reddit_label == 'elaboration': return 'B-COMMENT'
    if reddit_label == 'humor': return 'B-COMMENT'
    if reddit_label == 'other': return 'B-COMMENT'
    # if none of the conditions trip, 
    # the label is already correct, return itself
    return reddit_label

def generate_sdqc_mapping():
    return ['B-DENY', 'B-SUPPORT', 'B-QUERY', 'B-COMMENT']
    
def convert_label_sdqc_int2str(label_int):
    mappinglist = generate_sdqc_mapping()
    return mappinglist[label_int]

def convert_label_sdqc_str2int(label_str):
    mappinglist = generate_sdqc_mapping()
    return mappinglist.index(label_str)
    
def delete_deeper_levels(entries):
    '''
    Prunes the comments. Keeps only the threads that are 
    replying to the parent post

    Parameters
    ----------
    entries : list of dictionaries
        each list is a reddit thread.

    Returns 
    -------
    None
    '''
    for eachthread in entries:
        posts = eachthread['posts']
        firstpost_id = posts[0]['id']
        counter = 0
        while counter < len(posts):
            post = posts[counter]
            if post is posts[0]:
                # special case to allow first posts
                pass
            elif 'majority_link' not in post.keys():
                # post has no clear parent, remove it
                posts.remove(post)
                counter = counter - 1
            elif firstpost_id != post['majority_link']:
                # post's parent is not first post. remove
                posts.remove(post)
                counter = counter - 1
            counter = counter + 1

def clean_posts(entries):
    for eachthread in entries:
        posts = eachthread['posts']
        for post in posts:
            if 'body' not in post.keys():
                post['body'] = '[empty]'
            else:
                text = post['body']
                text = remove_urls(text)
                post['body'] = remove_spaces(text)
            
def store_as_sdqc_format(entries, directory, filename):
    tsvfile = open(directory+filename, 'w')
    tsvfile.write("index\t#1 Label\t#2 String\t#2 String\n")
    counter = 1
    #TODO save the length of the original threads
    #TODO modify the semeval code to do the same
    for eachthread in entries:
        posts_list = []
        label_list = []
        
        posts = eachthread['posts']
        for eachpost in posts:
            posts_list.append(eachpost['body'])
            strlabel = eachpost['majority_type']
            intlabel = convert_label_sdqc_str2int(strlabel)
            label_list.append(str(intlabel))
        
        s1 = ' ||||| '.join(posts_list)
        label = ','.join(label_list)
        tsvfile.write("%s\t%s\t%s\n" % (counter, label, s1))
        counter = counter + 1    
    
    tsvfile.close()
    
if __name__ == '__main__':
    FILEDIR  = './../data/coarse_discourse/'
    FILENAME = 'coarse_discourse_dump_reddit.json'
    
    tsvname = FILENAME.replace('.json', '.tsv')
    old_entries = extract_jsonfile(FILEDIR, FILENAME)
    mod_entries = extract_jsonfile(FILEDIR, FILENAME)
    eachthread = mod_entries[0]
    print(process_first_posts(mod_entries))
    convert_all_labels(mod_entries)
    plot_raw_statistics(old_entries)
    delete_deeper_levels(mod_entries)
    plot_raw_statistics(mod_entries)
    clean_posts(mod_entries)
    store_as_sdqc_format(mod_entries, FILEDIR, tsvname)
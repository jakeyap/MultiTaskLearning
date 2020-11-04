#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:20:34 2020
This script converts the coarse discourse and semeval files to the same style as the
semeval17 format, where each thread is a row and ' ||||| ' separates posts. 

However, there is a difference!!!
Each subsequent post is a reply to the previous post!!! e.g.
    post1 ||||| post1.1 ||||| post1.1.1

There is one key difference for the SemEval dataset 
Previously, it was 
    post1 ||||| post1.1 ||||| post1.2 ||||| post1.3...
    
Now, it will be
    post1 ||||| post1.1
    post1 ||||| post1.2
    post1 ||||| post1.3

@author: jakeyap
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocessor_functions import remove_urls, remove_spaces, post_isempty

from convert_coarse_discourse import extract_jsonfile

def map_label_2_int(label):
    label_arr = ['question',
                 'answer',
                 'announcement',
                 'agreement',
                 'appreciation',
                 'disagreement',
                 'negativereaction',
                 'elaboration',
                 'humor',
                 'other']
    return label_arr.index(label)

def map_int_2_label(int_label):
    label_arr = ['question',
                 'answer',
                 'announcement',
                 'agreement',
                 'appreciation',
                 'disagreement',
                 'negativereaction',
                 'elaboration',
                 'humor',
                 'other']
    return label_arr[int_label]

def clean_links(entries):
    '''
    Removes posts that don't have a majority_link
    Removes posts whose parents dont exist in the thread
    
    Parameters
    ----------
    entries : list of dictionaries
    
    Returns
    -------
    None.
    '''
    for eachthread in entries:
        posts = eachthread['posts']
        posts_len = len(posts)
        counter = 0
        while counter < posts_len:
            post = posts[counter]
            if 'is_first_post' not in post.keys():
                if 'majority_link' not in post.keys():
                    posts.remove(post)
                    counter -= 1
                    posts_len -= 1
            counter += 1
    
    for eachthread in entries:
        posts = eachthread['posts']
        potential_parents = set()
        first_post = posts[0]
        potential_parents.add(first_post['id'])
        for post in posts[1:]:
            
            index = posts.index(post)
            for i in range(index):
                potential_parent = posts[i]
                potential_parents.add(potential_parent['id'])
            if post['majority_link'] not in potential_parents:
                posts.remove(post)
    return

def clean_posts(entries):
    '''
    Removes whitespace and URLs from posts. Use title as body for empty posts

    Parameters
    ----------
    entries : list of dictionaries

    Returns 
    -------
    None
    '''
    for eachthread in entries:
        posts = eachthread['posts']
        for post in posts:
            if 'body' not in post.keys():
                text = remove_urls(eachthread['title'])
                post['body'] = remove_spaces(text)
            else:
                text = post['body']
                text = remove_urls(text)
                post['body'] = remove_spaces(text)
            if post_isempty(post['body']):
                text = remove_urls(eachthread['title'])
                post['body'] = remove_spaces(text)

def clean_labels_coarse_discourse(entries):
    '''
    Goes through coarse discourse dataset. For each post, if there are no annotator agreement, relabel as 'other'
    
    Parameters
    ----------
    entries : list of dictionaries
    
    Returns
    -------
    None.

    '''
    for eachthread in entries:
        posts = eachthread['posts']
        for each_post in posts:
            if 'majority_type' not in each_post.keys():
                # for posts with no annotator agreement, label as other
                each_post['majority_type'] = 'other'
    

def trace_root_2_leaves(thread):
    '''
    Goes through a thread and builds a list of lists
    
    Parameters
    ----------
    thread : a dictionary of a reddit thread

    Returns
    -------
    a tuple : (list_of_content, list_of_labels, orig_len)
    
    list_of_content : list of lists
        each list is a path from root to leaf.
    list_of_labels: list of lists
        each list contains labels corresponding to the post in content
        both list of lists have the same shape
    orig_len: int
        original size of the entire thread
    '''
    # summary of algo in this function
    # find the leaf nodes in this thread first
    # then for each leaf node
        # build 2 lists, 1 for actual posts, 1 for labels
        # append post's body and label to lists
        # append post's parent's body and label to lists
            # repeat until is_first_post
        # flip both whole lists
        # append the lists to list-of-lists
    
    list_of_content = []        # for storing post bodies
    list_of_labels = []         # for storing post labels
    
    posts = thread['posts']     # find all posts first
    orig_len = len(posts)       # store original full thread size
    
    # find leaf nodes first
    parent_ids = set()          # for storing all posts that are non-leaf
    
    for i in range(len(posts)):
        each_post = posts[i]
        if 0 == i:              # root post definitely is non-leaf
            parent_ids.add(each_post['id'])
        elif 'majority_link' in each_post.keys():   # for all other posts
            parent_id = each_post['majority_link']  # find parent id
            parent_ids.add(parent_id)               # add parent id to set of parents
    
    # finished building set of all non-leaf post ids
    for each_post in posts:             # go thru all posts in this thread
        post_id = each_post['id']       # find each post id
        if post_id not in parent_ids:   # if post is a leaf node
            content = []                # build 2 lists, 1 for actual posts
            labels = []                 # 1 for labels
            curr_post = each_post
            loop = True
            while (loop):
                if 'is_first_post' in curr_post.keys():
                    loop = False
                content.append(curr_post['body'])           # append post's body to lists
                labels.append(curr_post['majority_type'])   # append post's label to lists
                parent_id = curr_post['majority_link']      # get the parent
                curr_post_index = posts.index(curr_post)
                for candidate in posts[:curr_post_index]:
                    if parent_id == candidate['id']:
                        curr_post = candidate
            
            content.reverse()   # flip both lists
            labels.reverse()    # flip both lists
            
            list_of_content.append(content)
            list_of_labels.append(labels)
    
    if len(list_of_labels) == 0:
        # no posts in this thread, only root post exists. append it
        rootpost = posts[0]
        list_of_content.append([rootpost['body'],])
        list_of_labels.append([rootpost['majority_type'],])
    
    return (list_of_content, list_of_labels, orig_len)

def store_as_sdqc_format(thread_tuples, directory, filename):
    tsvfile = open(directory+filename, 'w')
    tsvfile.write("index\t#1 Label\t#2 Count\t#3 String\n")
    count = 1
    for eachthread in thread_tuples:
        list_of_posts, list_of_labels, orig_count = eachthread
        for index in range(len(list_of_labels)):
            posts_list = list_of_posts[index]
            label_list = []
            for each_label in list_of_labels[index]:
                intlabel = map_label_2_int(each_label)
                label_list.append(str(intlabel))
            
            s1 = ' ||||| '.join(posts_list)
            label = ','.join(label_list)
            tsvfile.write("%s\t%s\t%s\t%s\n" % (count, label, orig_count, s1))
            count = count + 1
    
    tsvfile.close()

if __name__ == '__main__':
    FILEDIR  = './../data/coarse_discourse/'
    FILENAME = 'coarse_discourse_dump_reddit.json'
    TESTFRACTION = 0.1
    
    test_tsvname = FILENAME.replace('.json', '_test_flat.tsv')
    dev_tsvname = FILENAME.replace('.json', '_dev_flat.tsv')
    train_tsvname = FILENAME.replace('.json', '_train_flat.tsv')
    
    old_entries = extract_jsonfile(FILEDIR, FILENAME)
    mod_entries = extract_jsonfile(FILEDIR, FILENAME)
    print('Cleaning unlinked posts')
    clean_links(mod_entries)
    print('Deleting posts with no clear labels')
    clean_labels_coarse_discourse(mod_entries)
    print('Cleaning whitespace and URLs')
    clean_posts(mod_entries)
    
    
    # eachthread = mod_entries[0]
    dataset_tuples = []
    counter = 0
    for each_thread in mod_entries:
        if counter % 10==0:
            print('Processing thread %d.' % counter)
        thread_tuple = trace_root_2_leaves(each_thread) # (list_of_content, list_of_labels, orig_len)
        if len(thread_tuple[1]) != 0:
            dataset_tuples.append(thread_tuple)
        counter = counter + 1
        
    # each tuple is a thread
    # split them into test, dev, train sets (10-10-80)
    
    np.random.seed(0)
    np.random.shuffle(dataset_tuples)           # shuffle threads first    
    num_thread = len(dataset_tuples)            # get the total count of threads
    test_idx = int(TESTFRACTION * num_thread)   # get the index for test set
    dev_idx = int(TESTFRACTION * num_thread*2)  # get the index for dev set
    
    test_data = dataset_tuples[0:test_idx]
    dev_data = dataset_tuples[test_idx:dev_idx]
    train_data = dataset_tuples[dev_idx:]
    
    store_as_sdqc_format(test_data, FILEDIR, test_tsvname)
    store_as_sdqc_format(dev_data, FILEDIR, dev_tsvname)
    store_as_sdqc_format(train_data, FILEDIR, train_tsvname)
    
    fig1, axes1 = plt.subplots(nrows=2,ncols=1,sharex=True)
    ax1 = axes1[0]
    ax2 = axes1[1]
    
    # count the original size of each tree
    tree_sizes = []
    
    for each_thread in train_data:
        treesize = each_thread[-1]  # each_thread is a tuple of (posts, labels, orig_len)
        tree_sizes.append(treesize)
    
    tree_depths = []
    for each_tuple in dev_data:
        list_of_labels = each_tuple[1]
        tree_depths.append(len(list_of_labels))
    for each_tuple in train_data:
        list_of_labels = each_tuple[1]
        tree_depths.append(len(list_of_labels))
    
    max_size = max(tree_sizes) + 1
    min_size = min(tree_sizes)
    med_size = torch.median(torch.tensor(tree_sizes)).item()
    
    ax1.hist(tree_sizes, bins=range(min_size,max_size), width=0.8)
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_title('Distn of training set tree sizes. Median=%d' % med_size)
    max_depth = max(tree_depths) + 1
    min_depth = min(tree_depths)
    med_depth = torch.median(torch.tensor(tree_depths))
    ax2.hist(tree_depths,bins=range(min_depth, max_depth), width=0.8)
    ax2.grid(True)
    ax2.set_title('Distn of training set tree depths. Median=%d' % med_depth)
    ax2.set_yscale('log')
    fig1.tight_layout()
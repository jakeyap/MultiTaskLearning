#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:46:29 2020

@author: jakeyap
"""
# TODO: working on this file now. Convert trees into dataloaders
# TODO: add a function to decode and print the tree

import torch
from transformers import BertTokenizer
import pandas as pd
import csv
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging

#import utilities.preprocessor_functions as preprocessor_functions
from utilities.handle_coarse_discourse_trees import RedditTree

logger = logging.getLogger(__name__)

default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
default_tokenizer.add_tokens(['[deleted]', '[URL]','[empty]'])

def reload_encoded_data(file='./data/coarse_discourse/full_trees/encoded_dict.pkl'):
    '''
    Reloads and returns tokenized+encoded pkl data

    Parameters
    ----------
    file : string of pkl filename. Default is 
    './data/coarse_discourse/full_trees/encoded_dict.pkl'.

    Returns
    -------
    dictionary where each key is post ID, value is encoded dictionary.
    '''
    return torch.load(file)

def loadfile(filename):
    logger.info('Loading ' + filename)
    return torch.load(filename)

def get_encoded_text_dict():
    '''
    Reloads and returns tokenized+encoded pkl data.
    dictionary where each key is post ID, value is encoded dictionary.
    '''
    return loadfile('./data/coarse_discourse/full_trees/encoded_dict.pkl')

def get_trees_test_set():
    '''  Returns list of RedditTree objects (test set) '''
    return loadfile('./data/coarse_discourse/full_trees/full_trees_test.pkl')


def get_trees_dev_set():
    ''' Returns list of RedditTree objects (dev set) '''
    return loadfile('./data/coarse_discourse/full_trees/full_trees_dev.pkl')

def get_trees_train_set():
    ''' Returns list of RedditTree objects (train set) '''
    return loadfile('./data/coarse_discourse/full_trees/full_trees_train.pkl')
        
def trees_2_df_approach_3(max_post_len, max_num_child, num_stride, root_trees, encoded_data):
    '''
    returns dataframe of training examples. trees are converted into examples using approach 3. 
    each root brings 3 or 4 kids, each kid brings 1 grandchild. stride at kid level
    each example is (for 3 kids)
    root, child1, grandchild1, child2, grandchild2, child3, grandchild3
    
    Parameters
    ----------
    max_post_len : int. 
        How many tokens to keep per post
    max_num_child : int. 
        How many kids to take per root post in 1 stride
    num_stride : int. 
        How many groups of kids to take
    root_trees : list 
        each element is a root tree
    encoded_data : dictionary 
        key-value-pairs of IDs and encoded data
    
    Returns
    -------
    dataframe with columns (post_id, input_ids, token_type_ids, attention_masks, labels_array, tree_size, fam_size)
    '''
    
    post_per_eg = max_num_child * 2 + 1     # num of posts = 1 root + n kids + n grandkids
    tensor_len = post_per_eg * max_post_len # tensor len = (num of posts) x (max len per post)
    
    list_of_post_ids  = []  # for storing all post ids of roots
    list_of_input_ids = []  # for storing all encoded input_ids
    list_of_type_ids  = []  # for storing all encoded token_type_ids
    list_of_att_masks = []  # for storing all attention_masks
    stance_arr_labels = []  # for storing all stance labels
    list_of_len_label = []  # for storing all tree size labels
    list_of_fam_label = []  # for storing all immediate family size labels
    
    for tree in root_trees:
        for i in range(num_stride):
            example = [tree]                        # list to store 1 example. element0 is root. 
            
            start = max_num_child * i               # start index for horz striding
            end = max_num_child * (i+1)             # end index for horz striding
            num_child = tree.num_child
            if num_child > (i * max_num_child):     # only enter the code to stride if the tree still has kids
                for kid in tree.children[start : end]:  # within a stride window
                    example.append(kid)                 # store child in window
                    if len(kid.children) != 0:          # if child has child, 
                        grand = kid.children[0]         # find grandkid
                        example.append(grand)           # store grandkid
                    else:                               # if child is childless
                        example.append('')              # store empty grandkid
                
                # For storing example's details across posts
                input_ids  = torch.zeros((1, tensor_len), dtype=torch.long)
                token_types= torch.zeros((1, tensor_len), dtype=torch.long)
                att_masks  = torch.zeros((1, tensor_len), dtype=torch.long)
                stance_labels = torch.ones((1, post_per_eg)) * -1
                
                # for every post in example, extract impt data, store in tensors
                for j in range (len(example)):
                    post = example[j]
                    if post != '':
                        post_id = post.post_id
                        enc_dict  = encoded_data[post_id]
                        input_id  = enc_dict['input_ids'].reshape((1,-1))
                        token_type= enc_dict['token_type_ids'].reshape((1,-1))
                        att_mask  = enc_dict['attention_mask'].reshape((1,-1))
                        
                        start = max_post_len * j
                        end = max_post_len * (j+1)
                        input_ids  [0,start:end] = input_id  [0,:max_post_len].detach().clone()
                        token_types[0,start:end] = token_type[0,:max_post_len].detach().clone()
                        att_masks  [0,start:end] = att_mask  [0,:max_post_len].detach().clone()
                        stance_labels[0,j] = post.label_int
                
                # store root's postid, all tensors into global list
                root = example[0]
                list_of_post_ids.append(root.post_id)
                list_of_input_ids.append(input_ids)
                list_of_type_ids.append(token_types)
                list_of_att_masks.append(att_masks)
                stance_arr_labels.append(stance_labels)
                list_of_len_label.append(root.tree_size)
                list_of_fam_label.append(root.num_child + root.num_grand)
    
    # convert lists into dataframe
    df = pd.DataFrame()
    df.insert(0, 'post_id', list_of_post_ids)
    df.insert(1, 'input_ids', list_of_input_ids)
    df.insert(2, 'token_type_ids', list_of_type_ids)
    df.insert(3, 'attention_masks', list_of_att_masks)
    df.insert(4, 'labels_array', stance_arr_labels)
    df.insert(5, 'tree_size', list_of_len_label)
    df.insert(6, 'fam_size', list_of_fam_label)
    return df

def build_trees_approach_4(max_post_len, max_num_child, encoded_data):
    ''' returns list of trees as built with approach 4 '''
    # TODO
    return

def get_df_strat_1(max_post_len=512,
                   strides=2,
                   DEBUG=False):
    """
    Gets trees encoded form, in dataframe format.

    Parameters
    ----------
    max_post_len : int, optional
        How many bert tokens to use per post. The default is 512.
    strides : int, optional
        How many horz strides to take. The default is 2.
    DEBUG : bool, optional
        If True, return only dev set
    Returns
    -------
    df_test : pandas dataframe 
        contains encoded tree info
    df_eval : pandas dataframe
        contains encoded tree info
    df_trng : pandas dataframe
        contains encoded tree info
    """
    if DEBUG:
        trees_test = get_trees_dev_set()
        trees_eval = get_trees_dev_set()
        trees_trng = get_trees_dev_set()
    else:
        trees_test = get_trees_test_set()
        trees_eval = get_trees_dev_set()
        trees_trng = get_trees_train_set()
    
    encoded_data = get_encoded_text_dict()  # bert tokenizer encoded data
    df_test = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=3, 
                                    num_stride=strides, 
                                    root_trees=trees_test, 
                                    encoded_data=encoded_data)
    
    df_eval = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=3, 
                                    num_stride=strides, 
                                    root_trees=trees_eval, 
                                    encoded_data=encoded_data)
    
    df_trng = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=3, 
                                    num_stride=strides, 
                                    root_trees=trees_trng, 
                                    encoded_data=encoded_data)
    
    return df_test, df_eval, df_trng

def get_df_strat_2(max_post_len=512,
                   strides=2):
    """
    Gets trees encoded form, in dataframe format.

    Parameters
    ----------
    max_post_len : int, optional
        How many bert tokens to use per post. The default is 512.
    strides : int, optional
        How many horz strides to take. The default is 2.

    Returns
    -------
    df_test : pandas dataframe 
        contains encoded tree info
    df_eval : pandas dataframe
        contains encoded tree info
    df_trng : pandas dataframe
        contains encoded tree info
    """
    trees_test = get_trees_test_set()
    trees_eval = get_trees_dev_set()
    trees_trng = get_trees_train_set()
    
    encoded_data = get_encoded_text_dict()  # bert tokenizer encoded data
    df_test = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=4, 
                                    num_stride=strides,
                                    root_trees=trees_test, 
                                    encoded_data=encoded_data)
    
    df_eval = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=4, 
                                    num_stride=strides, 
                                    root_trees=trees_eval, 
                                    encoded_data=encoded_data)
    
    df_trng = trees_2_df_approach_3(max_post_len=max_post_len, 
                                    max_num_child=4, 
                                    num_stride=strides, 
                                    root_trees=trees_trng, 
                                    encoded_data=encoded_data)
    
    return df_test, df_eval, df_trng

def df_2_dataloader(df, 
                    batchsize=64,
                    randomize=False,
                    DEBUG=False,
                    num_workers=0):
    """
    Converts dataframe into dataloaders

    Parameters
    ----------
    df : pandas dataframe
        {post_id, input_ids, token_type_ids, attention_masks, labels_array, tree_size, fam_size}.
    batchsize : int, optional
        minibatch size inside dataloaders. The default is 64.
    randomize : bool, optional
        flag to shuffle data or not. The default is False.
    DEBUG : bool, optional
        flag for debugging purposes. The default is False.
    num_workers : int, optional
        number of cores for loading data. The default is 0.

    Returns
    -------
    dataloader : pytorch dataloader object
        tuple with {post_index, input_ids, token_type_ids, attention_masks, labels_array, tree_size, fam_size}.
        index means index in the original dataframe
    """
    
    post_id = df.post_id.values                 # non numerical data cant be used in tensors. 
    post_id = post_id.reshape((-1,1))           # use index instead
    
    post_index = df.index.values                # numpy array. shape=(N,)
    post_index = torch.tensor(post_index)       # tensor, shape=(N,)
    post_index = post_index.reshape((-1,1))     # tensor, shape=(N,1)
    
    input_ids = df.input_ids.values.tolist()    # list, len=N. each element is a tensor, shape=(1,X).
    input_ids = torch.stack(input_ids, dim=0)   # tensor, shape=(N,1,X). X=num_posts x num_tokens
    input_ids = input_ids.squeeze(1)            # tensor, shape=(N,X)
    
    token_type_ids = df.token_type_ids.values.tolist()      # list, len=N
    token_type_ids = torch.stack(token_type_ids, dim=0)     # tensor, shape=(N,1,X)
    token_type_ids = token_type_ids.squeeze(1)              # tensor, shape=(N,X)
    
    attention_masks = df.attention_masks.values.tolist()    # list, len=N
    attention_masks = torch.stack(attention_masks, dim=0)   # tensor, shape=(N,1,X)
    attention_masks = attention_masks.squeeze(1)            # tensor, shape=(N,X)
    
    stance_labels = df.labels_array.values.tolist()         # list, len=N
    stance_labels = torch.stack(stance_labels, dim=0)       # tensor, shape=(N,1,num_posts)
    stance_labels = stance_labels.squeeze(1)                # tensor, shape=(N,num_posts)
    
    tree_size = torch.tensor(df.tree_size)                  # tensor, shape=(N,)
    tree_size = tree_size.reshape((-1,1))                   # tensor, shape=(N,1)
    
    fam_size = torch.tensor(df.fam_size)                    # tensor, shape=(N,)
    fam_size = fam_size.reshape((-1,1))                     # tensor, shape=(N,1)
    
    if DEBUG:
        dataset = TensorDataset(post_index[0:20],
                                input_ids[0:20],
                                token_type_ids[0:20],
                                attention_masks[0:20],
                                stance_labels[0:20],
                                tree_size[0:20],
                                fam_size[0:20])
    else:
        dataset = TensorDataset(post_index,
                                input_ids,
                                token_type_ids,
                                attention_masks,
                                stance_labels,
                                tree_size,
                                fam_size)
    if (randomize):
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batchsize,
                            num_workers=num_workers)
    
    return dataloader

if __name__ == '__main__':
    time1 = time.time()
    
    ''' Stuff for 20201103 - tokenize flattend reddit threads '''
    MAX_POST_PER_THREAD = 20
    MAX_POST_LENGTH = 256
    # DIRECTORY = './data/combined/'
    DIRECTORY = './data/coarse_discourse/'
    '''
    trees_test = get_trees_test_set()
    encoded_data = get_encoded_text_dict()
    
    df = trees_2_df_approach_3(max_post_len = 512, 
                               max_num_child = 3, 
                               num_stride = 2, 
                               root_trees = trees_test, 
                               encoded_data = encoded_data)
    '''
    df_test, df_eval, df_trng = get_df_strat_1(max_post_len=256,strides=2)
    dl_test = df_2_dataloader(df_test,batchsize=16,randomize=False,DEBUG=False,num_workers=4)
    dl_eval = df_2_dataloader(df_eval,batchsize=16,randomize=False,DEBUG=False,num_workers=4)
    dl_trng = df_2_dataloader(df_trng,batchsize=16,randomize=False,DEBUG=False,num_workers=4)
    
    print(len(df_test))
    print(len(df_eval))
    print(len(df_trng))
    time2 = time.time()
    minutes = (time2-time1) // 60
    seconds = (time2-time1) % 60
    print('%d minutes %2d seconds' % (minutes, seconds))
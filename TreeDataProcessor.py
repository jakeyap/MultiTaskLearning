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
    print('Loading ' + filename)
    return torch.load(filename)

def get_encoded_text_dict():
    '''
    Reloads and returns tokenized+encoded pkl data.
    dictionary where each key is post ID, value is encoded dictionary.
    '''
    return loadfile('./data/coarse_discourse/full_trees/encoded_dict.pkl')

def get_trees_test_set():
    ''' Returns list of RedditTree objects (test set) '''
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
    max_post_len : int. How many tokens to keep
    max_num_child : int. How many kids to take
    num_stride : int. How many groups of kids to take
    root_trees : list of root trees
    encoded_data : dict containing key-value-pairs of IDs and encoded data
    
    Returns
    -------
    dataframe with columns (post_id, input_ids, token_type_ids, attention_masks, labels_array, tree_size, fam_size)
    '''
    # TODO
    # get all trees
    # for each tree, build a list or itself, kids + grandkids
    # clone the encoded tensors over
    # 
    
    post_per_eg = max_num_child * 2 + 1 # 1 root, N kids, N grandkids
    tensor_len = post_per_eg * max_post_len
    
    list_of_post_ids  = []
    list_of_input_ids = []
    list_of_type_ids  = []
    list_of_att_masks = []
    stance_arr_labels = []
    list_of_len_label = []
    list_of_fam_label = []
    
    for tree in root_trees:
        for i in range(num_stride):
            example = [tree]    # list to store 1 example temporarily. element0 is root. 
            
            start = max_num_child * i
            end = max_num_child * (i+1)
            
            for kid in tree.children[start : end]:
                example.append(kid)
                if len(kid.children) != 0:
                    grand = kid.children[0]
                    example.append(grand)
                else:
                    example.append('')
            
            # a list of posts constructed. get encoded data
            input_ids  = torch.zeros((1, tensor_len))
            token_types= torch.zeros((1, tensor_len))
            att_masks  = torch.zeros((1, tensor_len))
            stance_labels = torch.ones((1, post_per_eg)) * -1
            
            print(example)
            for j in range (len(example)):
                post = example[j]
                if post != '':
                    post_id = post.post_id
                    enc_dict  = encoded_data[post_id]
                    input_id  = enc_dict['input_ids'].reshape((1,-1))
                    token_type= enc_dict['token_type_ids'].reshape((1,-1))
                    att_mask  = enc_dict['attention_mask'].reshape((1,-1))
                    
                    print('post id  '+post_id)
                    print('input id '+str(input_id.shape))
                    print('token_ty '+str(token_type.shape))
                    print('att mask '+str(att_mask.shape))
                    
                    start = max_post_len * j
                    end = max_post_len * (j+1)
                    input_ids  [0,start:end] = input_id  [0,:max_post_len].detach().clone()
                    token_types[0,start:end] = token_type[0,:max_post_len].detach().clone()
                    att_masks  [0,start:end] = att_mask  [0,:max_post_len].detach().clone()
                    stance_labels[0,j] = post.label_int
            
            root = example[0]
            list_of_post_ids.append(root.post_id)
            list_of_input_ids.append(input_ids)
            list_of_type_ids.append(token_types)
            list_of_att_masks.append(att_masks)
            stance_arr_labels.append(stance_labels)
            list_of_len_label.append(root.tree_size)
            list_of_fam_label.append(root.num_child + root.num_grand)
            
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

def assemble_tree_strat3(max_post_len, max_num_posts):
    ''' returns the tree as built with strategy 3 '''
    # TODO
    return


def tokenize_encode_thread(thread, max_post_length, max_post_per_thread):
    '''
    Tokenizes and encodes a thread.

    Parameters
    ----------
    thread : list of strings
        each string is a reddit or twitter post 
    max_post_length : int
        maximum number of tokens per post to procecss
    max_post_per_thread : int
        number of posts per threads to work on

    Returns
    -------
    encoded_posts : list with length==max_post_per_thread
        Each element is a tensor. Matches a post after tokenizing/encoding
    token_type_ids : list with length==max_post_per_thread
        Each element is a tensor. Matches a post after tokenizing/encoding
    attention_masks : list with length==max_post_per_thread
        Each element is a tensor. Matches a post after tokenizing/encoding

    '''
    #encoded_posts = []
    #token_type_ids = []
    #attention_masks = []
    counter = 0
    #encoded_dict = tokenize_encode_post(thread[counter], max_post_length)
    encoded_dict=None
    encoded_posts = encoded_dict['input_ids']
    token_type_ids = encoded_dict['token_type_ids']
    attention_masks = encoded_dict['attention_mask']
    
    counter = 1
    while (counter < max_post_per_thread):
        try:
            #encoded_dict = tokenize_encode_post(thread[counter], max_post_length)
            encoded_dict = None
            temp_encoded_post = encoded_dict['input_ids']
            temp_token_type_ids = encoded_dict['token_type_ids']
            temp_attention_masks = encoded_dict['attention_mask']
            encoded_posts = torch.cat((encoded_posts, temp_encoded_post), dim=1)
            token_type_ids = torch.cat((token_type_ids, temp_token_type_ids), dim=1)
            attention_masks = torch.cat((attention_masks, temp_attention_masks), dim=1)
            
        except IndexError:
            encoded_posts = torch.cat((encoded_posts, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
            token_type_ids = torch.cat((token_type_ids, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
            attention_masks = torch.cat((attention_masks, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
        counter = counter + 1
    return encoded_posts, token_type_ids, attention_masks

def tokenize_encode_dataframe(dataframe, max_post_length, max_post_per_thread):
    logging.info('Tokenizing & encoding started.')
    logging.info('\tPosts per thread: %d' % max_post_per_thread)
    logging.info('\tMax tokens per post: %d' % max_post_length)
    
    list_of_encoded_comments = []
    list_of_token_type_ids = []
    list_of_attention_masks = []
    list_of_labels_array = []
    
    for i in range(len(dataframe)):
        if i % 100 == 0:
            print('Processing line %d' % i)
        thread = dataframe.text[i]
        temp = tokenize_encode_thread(thread=thread, 
                                      max_post_length = max_post_length, 
                                      max_post_per_thread = max_post_per_thread)
        list_of_encoded_comments.append(temp[0])
        list_of_token_type_ids.append(temp[1])
        list_of_attention_masks.append(temp[2])
        
        labels_list = dataframe.labels_list[i]
        labels_arr  = convert_label_list_2_tensor(labels_list=labels_list, 
                                                  max_post_per_thread=max_post_per_thread)
        list_of_labels_array.append(labels_arr)
    
    width = dataframe.shape[1]
    dataframe.insert(width+0, 'encoded_comments', list_of_encoded_comments)
    dataframe.insert(width+1, 'token_type_ids', list_of_token_type_ids)
    dataframe.insert(width+2, 'attention_masks', list_of_attention_masks)
    dataframe.insert(width+3, 'labels_array', list_of_labels_array)
    return dataframe

def convert_label_list_2_tensor(labels_list, max_post_per_thread):
    '''
    Takes a list of text labels and converts into a tensor.
    If there are less entries than the stiuplated rows, -1 is entered
    eg. ['1', '2', '3'] becomes [1 ,2, 3, -1]
    Parameters
    ----------
    label_list : list of text labels
        type labels for each thread.
    max_post_per_thread : int
        maximum length per thread to look at

    Returns
    -------
    labels_tensor : TYPE
        DESCRIPTION.

    '''
    labels_tensor = -1 * torch.ones(size=(1,max_post_per_thread))
    counter = 0
    for eachlabel in labels_list:
        if counter >= max_post_per_thread:
            break
        else:
            labels_tensor[0,counter] = int(eachlabel)
        counter = counter + 1
    return labels_tensor


def dataframe_2_dataloader(dataframe, 
                           batchsize=64,
                           randomize=False,
                           DEBUG=False,
                           num_workers=0):
    '''
    

    Parameters
    ----------
    dataframe : pandas dataframe with these columns 
        {orig_length,
        labels_list,
        text,
        encoded_comments,
        token_type_ids,
        attention_masks}
    batchsize : int, minibatch size inside dataloaders. Defaults to 64.
    randomize : boolean flag to shuffle data or not. Defaults to False.
    DEBUG : boolean flag for debugging purposes. Defaults to False. Unused
    num_workers : int, number of cores for loading data. Defaults to 0.

    Returns
    -------
    dataloader : pytorch dataloader type
        Each dataloader is packed into the following tuple
            {index in original data,
            tensor of encoded_comments, 
            tensor of token_typed_ids,
            tensor of attention_masks,
            original true length label
            tensor of true stance labels}
    '''
    posts_index     = dataframe.index.values
    posts_index     = posts_index.reshape((-1,1))
    
    encoded_comments = dataframe['encoded_comments'].values.tolist()
    encoded_comments = torch.stack(encoded_comments, dim=0).squeeze(1)
    
    token_type_ids  = dataframe['token_type_ids'].values.tolist()
    token_type_ids  = torch.stack(token_type_ids, dim=0).squeeze(1)
    
    attention_masks = dataframe['attention_masks'].values.tolist()
    attention_masks = torch.stack(attention_masks, dim=0).squeeze(1)
    
    orig_length     = dataframe['orig_length'].values.reshape((-1,1))
    
    stance_labels   = dataframe['labels_array'].tolist()
    stance_labels   = torch.stack(stance_labels, dim=0).squeeze(1)
        
    posts_index = torch.from_numpy(posts_index)
    orig_length = torch.from_numpy(orig_length)
    if DEBUG:
        dataset = TensorDataset(posts_index[0:40],
                                encoded_comments[0:40],
                                token_type_ids[0:40],
                                attention_masks[0:40],
                                orig_length[0:40],
                                stance_labels[0:40])
    else:
        dataset = TensorDataset(posts_index,
                                encoded_comments,
                                token_type_ids,
                                attention_masks,
                                orig_length,
                                stance_labels)
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
    trees_test = get_trees_test_set()
    encoded_data = get_encoded_text_dict()
    
    df = trees_2_df_approach_3(max_post_len = 512, 
                               max_num_child = 3, 
                               num_stride = 2, 
                               root_trees = trees_test, 
                               encoded_data = encoded_data)
    time2 = time.time()
    minutes = (time2-time1) // 60
    seconds = (time2-time1) % 60
    print('%d minutes %2d seconds' % (minutes, seconds))
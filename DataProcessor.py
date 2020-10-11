# -*- coding: utf-8 -*-

"""
Created on Mon Sep 14 2020 11:19 

@author: jakeyap
"""

import torch
from transformers import BertTokenizer
import pandas as pd
import csv
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
default_tokenizer.add_tokens(['[deleted]', '[URL]','[empty]'])

def open_tsv_data(filename):
    '''
    Reads TSV data. Outputs a list of list where
        [
            [index, labels, original_length, comments]
            [index, labels, original_length, comments]
            ...
        ]

    Parameters
    ----------
    filename : string
        string that contains filename of TSV file

    Returns
    -------
    lines : list of list (See above)
    '''
    with open(filename, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        examples = raw_text_to_examples(lines)
        return examples

def raw_text_to_examples(tsv_lines):
    '''
    Converts raw text from tsv file into pandas dataframe

    Parameters
    ----------
    tsv_lines : list of list
        Outer list length is the number of entries in tsv file.
        Inner list length is length 4, [index, orig_length, labelslist, posts]

    Returns
    -------
    examples : pandas dataframe
        Contains all the TSV stuff into a dataframe

    '''
    counter = 0
    examples = []
    # remember to skip headers
    for eachline in tsv_lines[1:]:
        labels_list = eachline[1].split(',')
        orig_length = int(eachline[2])
        text = eachline[3].lower().split(' ||||| ')
        examples.append({'orig_length':orig_length, 
                         'labels_list':labels_list, 
                         'text':text})
        counter = counter + 1
    examples = pd.DataFrame(examples)
    return examples

def get_dataset(filename):
    return open_tsv_data(filename)

def get_test_set_shuffled():
    return get_dataset('./data/combined/shuffled_test.tsv')

def get_dev_set_shuffled():
    return get_dataset('./data/combined/shuffled_dev.tsv')

def get_train_set_shuffled():
    return get_dataset('./data/combined/shuffled_train.tsv')

def get_test_set():
    return get_dataset('./data/combined/combined_test.tsv')

def get_dev_set():
    return get_dataset('./data/combined/combined_dev.tsv')

def get_train_set():
    return get_dataset('./data/combined/combined_train.tsv')
        
def tokenize_encode_post(post, max_post_length):
    tokens = default_tokenizer.tokenize(post)
    return default_tokenizer.__call__(text=tokens, 
                                      padding='max_length',
                                      truncation=True,
                                      is_pretokenized=True,
                                      max_length=max_post_length,
                                      return_tensors='pt')

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
    
    encoded_dict = tokenize_encode_post(thread[counter], max_post_length)
    encoded_posts = encoded_dict['input_ids']
    token_type_ids = encoded_dict['token_type_ids']
    attention_masks = encoded_dict['attention_mask']
    
    counter = 1
    while (counter < max_post_per_thread):
        try:
            encoded_dict = tokenize_encode_post(thread[counter], max_post_length)
            temp_encoded_post = encoded_dict['input_ids']
            temp_token_type_ids = encoded_dict['token_type_ids']
            temp_attention_masks = encoded_dict['attention_mask']
            encoded_posts = torch.cat((encoded_posts, temp_encoded_post), dim=1)
            token_type_ids = torch.cat((token_type_ids, temp_token_type_ids), dim=1)
            attention_masks = torch.cat((attention_masks, temp_attention_masks), dim=1)
            
            #encoded_posts.append(encoded_dict['input_ids'])
            #token_type_ids.append(encoded_dict['token_type_ids'])
            #attention_masks.append(encoded_dict['attention_mask'])
        except IndexError:
            #encoded_posts.append(torch.zeros(size=(1, max_post_length),dtype=torch.int64))
            #token_type_ids.append(torch.zeros(size=(1, max_post_length),dtype=torch.int64))
            #attention_masks.append(torch.zeros(size=(1, max_post_length),dtype=torch.int64))
            encoded_posts = torch.cat((encoded_posts, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
            token_type_ids = torch.cat((token_type_ids, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
            attention_masks = torch.cat((attention_masks, torch.zeros(size=(1, max_post_length),dtype=torch.int64)), dim=1)
        counter = counter + 1
    return encoded_posts, token_type_ids, attention_masks

def tokenize_encode_dataframe(dataframe, max_post_length, max_post_per_thread):
    print('Tokenizing & encoding started.')
    print('\tPosts per thread: %d' % max_post_per_thread)
    print('\tMax tokens per post: %d' % max_post_length)
    
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

def save_df_2_pkl(dataframe, filename):
    print('saving to pickle file ' + filename)
    torch.save(dataframe, filename)
    #dataframe.to_pickle(filename)

def load_df_from_pkl(filename):
    print('loading from pickle file')
    return torch.load(filename)
    #return pd.read_pickle(filename)

def dataframe_2_dataloader(dataframe, 
                           batchsize=64,
                           randomize=False,
                           DEBUG=True):
    
    '''
    Each dataframe has the following columns 
    {
        orig_length
        labels_list
        text
        encoded_comments
        token_type_ids
        attention_masks
    }
    
    Each dataloader is packed into the following tuple
    {   
         index in original data,
         tensor of encoded_comments, 
         tensor of token_typed_ids,
         tensor of attention_masks,
         original true length label
         tensor of true stance labels
    }
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
                            batch_size=batchsize)
    
    return dataloader

if __name__ == '__main__':
    MAX_POST_PER_THREAD = 4
    MAX_POST_LENGTH = 512
    DIRECTORY = './data/combined/'
    
    filenames = ['shuffled_dev', 'shuffled_test', 'shuffled_train', 
                 'combined_dev', 'combined_test', 'combined_train']
    
    time1 = time.time()
    for each_filename in filenames:
        print('Encoding dataset: ' + each_filename)
        suffix = '_' + str(MAX_POST_PER_THREAD) + '_' +str(MAX_POST_LENGTH)
        tsv_filename = DIRECTORY + each_filename +'.tsv'
        pkl_filename = DIRECTORY + 'encoded_' + each_filename + suffix +'.pkl'
        dataframe = get_dataset(tsv_filename)
        dataframe = tokenize_encode_dataframe(dataframe, MAX_POST_LENGTH, MAX_POST_PER_THREAD)
        save_df_2_pkl(dataframe, pkl_filename)
    time2 = time.time()
    time_taken = int(time2-time1)
    print('time taken: %ds' % time_taken)
    

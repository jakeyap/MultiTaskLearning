#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:07:44 2020

@author: jakeyap
"""

import matplotlib.pyplot as plt
import torch
import DataProcessor
import multitask_helper_functions as helper
from classifier_models import my_ModelA0, my_ModelBn, my_ModelDn

def accuracy_from_recall(short_recall, short_support, long_recall, long_support):
    short_true_pos = short_support * short_recall
    long_true_pos = long_support * long_recall
    true_pos = short_true_pos + long_true_pos
    total_samples = short_support + long_support
    return true_pos / total_samples

def reload_modelA0(model_filename, 
                   MAX_POST_PER_THREAD=4, 
                   MAX_POST_LENGTH=256):
    # For reviving a previously trained modelA0
    model = my_ModelA0.from_pretrained('bert-base-uncased',
                                       stance_num_labels=5,
                                       length_num_labels=2,
                                       max_post_num=MAX_POST_PER_THREAD, 
                                       max_post_length=MAX_POST_LENGTH)
    
    token_len = len(DataProcessor.default_tokenizer)
    model.resize_token_embeddings(token_len)    # Resize model vocab
    temp = torch.load(model_filename)           # Reload model and optimizer states
    state_dict = temp[0]                        # Extract state dict
    # stance_opt_state = temp[1]
    # length_opt_state = temp[2]
    model.load_state_dict(state_dict)           # Stuff state into model
    return model

def reload_modelBn(model_filename,
                   number,
                   MAX_POST_PER_THREAD=4,
                   MAX_POST_LENGTH=256):
    # For reviving a previously trained modelBn
    model = my_ModelBn.from_pretrained('bert-base-uncased',
                                       stance_num_labels=5,
                                       length_num_labels=2,
                                       max_post_num=MAX_POST_PER_THREAD, 
                                       max_post_length=MAX_POST_LENGTH,
                                       num_transformers=number)
    token_len = len(DataProcessor.default_tokenizer)
    model.resize_token_embeddings(token_len)    # Resize model vocab
    temp = torch.load(model_filename)           # Reload model and optimizer states
    state_dict = temp[0]                        # Extract state dict
    model.load_state_dict(state_dict)           # Stuff state into model
    return model

def reload_modelDn(model_filename,
                   number,
                   MAX_POST_PER_THREAD=4,
                   MAX_POST_LENGTH=256):
    model = my_ModelDn.from_pretrained('bert-base-uncased', 
                                        stance_num_labels=11,
                                        length_num_labels=2,
                                        max_post_num=8,
                                        max_post_length=256,
                                        exposed_posts=4,
                                        num_transformers=number)
    token_len = len(DataProcessor.default_tokenizer)
    model.resize_token_embeddings(token_len)    # Resize model vocab
    temp = torch.load(model_filename)           # Reload model and optimizer states
    try:
        state_dict = temp[0]                    # Extract state dict
        model.load_state_dict(state_dict)       # Stuff state into model
    except Exception:
        state_dict = temp                       # Extract state dict
        model.load_state_dict(state_dict)       # Stuff state into model
    return model

def reload_test_data(test_filename = 'encoded_shuffled_test_4_256.pkl', BATCH_SIZE_TEST=2):
    # Grab test data pickle file and reload into a dataframe
    test_dataframe = DataProcessor.load_from_pkl(test_filename)
    test_dataloader = DataProcessor.dataframe_2_dataloader(test_dataframe,
                                                           batchsize=BATCH_SIZE_TEST,
                                                           randomize=False,
                                                           DEBUG=False,
                                                           num_workers=1)
    return test_dataloader

def replot_training_losses(filename):
    # For plotting training loss and f1 scores
    
    lossfile = torch.load(filename)     # Open the loss file
    train_stance_loss = lossfile[0]     # Expand the training data
    train_length_loss = lossfile[1]
    train_horz_index  = lossfile[2]
    
    valid_stance_loss = lossfile[3]     # Expand the dev set data
    valid_length_loss = lossfile[4]
    valid_f1_scores   = lossfile[5]    
    valid_horz_index  = lossfile[6]
    
    fig,axes = plt.subplots(3,1)        # Create 3 plots, reference them
    fig.show()
    ax1,ax2,ax3 = axes[0],axes[1],axes[2]
    
    ax1.set_title('stance loss')        # Plot stance losses
    ax1.scatter(train_horz_index, train_stance_loss, color='red', s=10)
    ax1.scatter(valid_horz_index, valid_stance_loss, color='blue')
    ax1.set_yscale('linear')
    
    ax2.set_title('length loss')        # Plot length losses
    ax2.scatter(train_horz_index, train_length_loss, color='red', s=10)
    ax2.scatter(valid_horz_index, valid_length_loss, color='blue')
    ax2.set_yscale('linear')
    
    ax3.set_title('f1 score')
    ax3.scatter(valid_horz_index, valid_f1_scores, color='red')
    #ax3.scatter(valid_horz_index[-1], test_f1_score, color='blue')
    ax3.set_xlabel('minibatches')
    xlim = ax2.get_xlim()   # for setting the graphs x-axis to be the same
    ax3.set_xlim(xlim)      # for setting the graphs x-axis to be the same
    for each_axis in axes:
        each_axis.grid(True)
        each_axis.grid(True,which='minor')
    
    fig.set_size_inches(6, 8)
    fig.tight_layout()
    
    # Return the lossfile and figure handles
    return lossfile, fig, axes

if __name__ =='__main__':
    filename = './log_files/training_losses/losses_ModelD4_exp43.bin'
    losses, fig, axes = replot_training_losses(filename)
    
    model_filename = './saved_models/ModelD4_exp43.bin'
    #model = reload_modelA0(model_filename,4,256)
    model = reload_modelDn(model_filename, 
                           number=4,
                           MAX_POST_PER_THREAD=4,
                           MAX_POST_LENGTH=256)
    model.eval()
    model.cuda()
    test_dataframe = DataProcessor.load_from_pkl('./data/coarse_discourse/encoded_coarse_discourse_dump_reddit_test_flat_20_256.pkl')
    
    index_2_choose = 10
    i = index_2_choose
    encoded_comments = test_dataframe['encoded_comments'][i]
    token_type_ids = test_dataframe['token_type_ids'][i]
    attention_masks = test_dataframe['token_type_ids'][i]
    length_labels = test_dataframe['orig_length'][i]
    stance_labels = test_dataframe['labels_array'][i]
    
    MAX_POST_PER_THREAD = 4
    # keep the ones needed only
    length_labels = length_labels[:,0:MAX_POST_PER_THREAD]
    stance_labels = stance_labels[:,0:MAX_POST_PER_THREAD]
    gpu = 'cuda'
    # move features to gpu
    encoded_comments = encoded_comments.to(gpu)
    token_type_ids = token_type_ids.to(gpu)
    attention_masks = attention_masks.to(gpu)
    length_labels = length_labels.to(gpu)
    stance_labels = stance_labels.to(gpu)
    '''
    encoded_comments = minibatch[1] # shape = (n, 20xMAX_POST_LENGTH), will be truncated inside model
    token_type_ids = minibatch[2]   # shape = (n, 20xMAX_POST_LENGTH), will be truncated inside model
    attention_masks = minibatch[3]  # shape = (n, 20xMAX_POST_LENGTH), will be truncated inside model
    length_labels = minibatch[4]    # shape = (n, 20) need to truncate here
    stance_labels = minibatch[5]    # shape = (n, 20) need to truncate here
    '''
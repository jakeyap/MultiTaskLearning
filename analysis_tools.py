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
from classifier_models import my_ModelA0, my_ModelBn

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

def reload_test_data(test_filename = 'encoded_shuffled_test_4_256.pkl', BATCH_SIZE_TRAIN=2):
    # Grab test data pickle file and reload into a dataframe
    test_dataframe = DataProcessor.load_from_pkl(test_filename)
    test_dataloader = DataProcessor.dataframe_2_dataloader(test_dataframe,
                                                           batchsize=BATCH_SIZE_TRAIN,
                                                           randomize=False,
                                                           DEBUG=False,
                                                           num_workers=1)
    return test_dataloader

def test(model, dataloader, MAX_POST_PER_THREAD=4):
    model.eval()        # Set network into evaluation mode
    
    # Start the arrays to store the entire test set. 
    # Initialize a blank 1st data point first. Remember to delete later    
    stance_logits_arr = torch.zeros(size=(1,MAX_POST_PER_THREAD,5),dtype=torch.float)
    stance_labels_arr = torch.zeros(size=(1,MAX_POST_PER_THREAD),dtype=torch.int64)
    length_logits_arr = torch.zeros(size=(1,2),dtype=torch.float)
    length_labels_arr = torch.zeros(size=(1,1),dtype=torch.int64)
    
    with torch.no_grad():
        for batch_idx, minibatch in enumerate(dataloader):
            print('\tTesting test set Minibatch: %4d' % batch_idx)
            # get the features from dataloader
            # posts_index = minibatch[0]
            encoded_comments = minibatch[1]
            token_type_ids = minibatch[2]
            attention_masks = minibatch[3]
            length_labels = minibatch[4]
            stance_labels = minibatch[5]
            
            # move features to gpu
            gpu = 'cuda'
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            length_labels = length_labels.to(gpu)
            stance_labels = stance_labels.to(gpu)
            
            stance_logits = model(input_ids = encoded_comments,         # get the stance prediction logits
                                  token_type_ids = token_type_ids,      # (n,A,B): n=minibatch, A=max_posts_per_thread, B=num of classes
                                  attention_masks = attention_masks, 
                                  task='stance')
            length_logits = model(input_ids = encoded_comments,         # get the length prediction logits
                                  token_type_ids = token_type_ids,      # (n,2): n=minibatch, 2=num of classes
                                  attention_masks = attention_masks, 
                                  task='length')
            
            stance_logits_arr = torch.cat((stance_logits_arr,           # store stance logits in a big linear array (N,A,B)
                                           stance_logits.to('cpu')),
                                          dim=0)
            length_logits_arr = torch.cat((length_logits_arr,           # store length logits in a big linear array (N,2)
                                           length_logits.to('cpu')),
                                          dim=0)
            
            stance_labels_arr = torch.cat((stance_labels_arr,           # store all stance labels in a big linear array (NA,1)
                                           stance_labels.to('cpu').long()),
                                          dim=0)
            length_labels_arr = torch.cat((length_labels_arr,           # store all length labels in a big linear array (N,1))
                                           length_labels.to('cpu').long()),
                                          dim=0)
            
        # Discarding the blank 1st data point.
        stance_logits_arr = stance_logits_arr[1:,:,:]   # shape was (n+1,A,5)
        stance_labels_arr = stance_labels_arr[1:,:,]    # shape was (n+1,A)
        length_logits_arr = length_logits_arr[1:,:]     # shape was (n+1,2)
        length_labels_arr = length_labels_arr[1:,:]     # shape was (n+1,1)
        
        stance_loss = helper.stance_loss(stance_logits_arr.to(gpu),     # calculate the dev set stance loss
                                         stance_labels_arr.to(gpu),     # move to GPU, cauz loss weights are in GPU
                                         loss_fn=stance_loss_fn)
        
        length_loss = helper.length_loss(pred_logits=length_logits_arr, # calculate the dev set length loss
                                         true_labels=length_labels_arr, 
                                         loss_fn=length_loss_fn,
                                         divide=THREAD_LENGTH_DIVIDER)
        
        # convert everything into linear tensors
        stance_pred = helper.logit_2_class_stance(stance_logits_arr)    # convert logits to stance labels. (nA,)
        length_pred = helper.logit_2_class_length(length_logits_arr)    # convert logits to length labels. (n,)
        stance_true = stance_labels_arr.reshape(stance_pred.shape)      # reshape from (n,A) into (nA,)
        length_true = length_labels_arr.reshape(length_pred.shape)      # reshape from (n,1) into (n,)
        
        stance_metrics = helper.stance_f1(stance_pred,                  # calculate the f1-metrics for stance
                                          stance_true)
        length_metrics = helper.length_f1(length_pred,                  # calculate the f1-metrics for length
                                          length_true,
                                          THREAD_LENGTH_DIVIDER)
        
        stance_accuracy = helper.accuracy_stance(stance_pred,           # calculate prediction accuracies
                                                 stance_true)
        length_accuracy = helper.accuracy_length(length_pred,           # calculate prediction accuracies
                                                 length_true,
                                                 THREAD_LENGTH_DIVIDER)
        
        stance_msg = helper.stance_f1_msg(stance_metrics[0],            # Get the strings to display for f1 scores
                                          stance_metrics[1],
                                          stance_metrics[2],
                                          stance_metrics[3],
                                          stance_metrics[4])
        length_msg = helper.length_f1_msg(length_metrics[0],            # Get the strings to display for f1 scores
                                          length_metrics[1],
                                          length_metrics[2],
                                          length_metrics[3],
                                          length_metrics[4])
        
        f1_stance_macro = stance_metrics[4]
        f1_length_macro = length_metrics[4]
        
        print('\n'+stance_msg)
        print('Stance Accuracy: %1.4f' % stance_accuracy)
        print('\n'+length_msg)
        print('Length Accuracy: %1.4f' % length_accuracy)
    
    return stance_pred, stance_true, length_pred, length_true, \
            stance_loss.item(), length_loss.item(), \
            f1_stance_macro, f1_length_macro, \
            stance_accuracy, length_accuracy

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
    filename = './log_files/losses_25_10_4_256_0.0002_exp6.bin'
    losses, fig, axes = replot_training_losses(filename)
    
    model_filename = './saved_models/ModelA0_25_10_4_256_0.0002_exp6.bin'
    model = reload_modelA0(model_filename,4,256)
    model.eval()
    
    
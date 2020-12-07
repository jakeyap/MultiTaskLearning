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
from classifier_models import my_ModelA0, my_ModelBn, my_ModelDn, my_ModelEn
import numpy as np
import time

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

def reload_modelEn(model_filename,
                   number,
                   MAX_POST_PER_THREAD=4,
                   MAX_POST_LENGTH=256):
    model = my_ModelEn.from_pretrained('bert-base-uncased',
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

def reload_train_dataframe(train_filename = './data/coarse_discourse/encoded_coarse_discourse_dump_reddit_train_flat_20_256.pkl'):
    train_dataframe = DataProcessor.load_from_pkl(train_filename)   # Grab test data pickle file, put into a dataframe
    return train_dataframe

def reload_test_dataframe(test_filename = './data/coarse_discourse/encoded_coarse_discourse_dump_reddit_test_flat_20_256.pkl'):
    test_dataframe = DataProcessor.load_from_pkl(test_filename)     # Grab test data pickle file, put into a dataframe
    return test_dataframe

def df2dataloader(dataframe, BATCH_SIZE=2):
    dataloader = DataProcessor.dataframe_2_dataloader(dataframe,
                                                      batchsize=BATCH_SIZE,
                                                      randomize=False,
                                                      DEBUG=False,
                                                      num_workers=4)
    return dataloader

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

def convert_label_2_string(number):
    label_ind = ['question',    'answer', 
                 'announcement','agreement',
                 'appreciation','disagreement',
                 'negativereaction',
                 'elaboration', 'humor',
                 'other']
    if number != -1:
        return label_ind[int(number)][0:4]
    else:
        return 'none'

def reprocess_df():
    ''' 
    adds in the labels for doing the error analysis plot
    for eg. 'ques-answ-nega' will be appended for a sequence of posts with 
    question-answer-negativereaction tags.
    
    there will be 4 labels appended.
    1st: only 1 label
    2nd: up to 2 labels
    3rd: up to 3 labels
    4th: up to 4 labels
    '''
    train_df = reload_train_dataframe()
    # attach labels 
    fulllabels = train_df.labels_array
    label3 = []
    label2 = []
    label1 = []
    label0 = []
    
    label_seq_4 = []
    label_seq_3 = []
    label_seq_2 = []
    label_seq_1 = []
    
    for eachtensor in fulllabels:
        array = eachtensor.numpy()
        label3.append(array[0,3])
        label2.append(array[0,2])
        label1.append(array[0,1])
        label0.append(array[0,0])
        
        label_seq = convert_label_2_string(array[0,0])
        label_seq_1.append(label_seq)
        label_seq += '-' + convert_label_2_string(array[0,1])
        label_seq_2.append(label_seq)
        label_seq += '-' + convert_label_2_string(array[0,2])
        label_seq_3.append(label_seq)
        label_seq += '-' + convert_label_2_string(array[0,3])
        label_seq_4.append(label_seq)
    
    col_length = len(train_df.columns)
    train_df.insert(loc=col_length, column='label_seq_4',value=label_seq_4)
    train_df.insert(loc=col_length, column='label_seq_3',value=label_seq_3)
    train_df.insert(loc=col_length, column='label_seq_2',value=label_seq_2)
    train_df.insert(loc=col_length, column='label_seq_1',value=label_seq_1)
    
    train_df.insert(loc=col_length, column='label3',value=label3)
    train_df.insert(loc=col_length, column='label2',value=label2)
    train_df.insert(loc=col_length, column='label1',value=label1)
    train_df.insert(loc=col_length, column='label0',value=label0)
    
    return train_df
    
def inspect_labels_vs_length():
    df = reprocess_df()
    
    seq_sequences_1 = df.label_seq_1
    seq_sequences_2 = df.label_seq_2
    seq_sequences_3 = df.label_seq_3
    seq_sequences_4 = df.label_seq_4
    
    # count number of unique sequences
    unique_seq_1 = set(seq_sequences_1)
    unique_seq_2 = set(seq_sequences_2)
    unique_seq_3 = set(seq_sequences_3)
    unique_seq_4 = set(seq_sequences_4)
    
    print('Number of unique 1-label sequences: %d ' % len(unique_seq_1))
    print('Number of unique 2-label sequences: %d ' % len(unique_seq_2))
    print('Number of unique 3-label sequences: %d ' % len(unique_seq_3))
    print('Number of unique 4-label sequences: %d ' % len(unique_seq_4))
    
    # generate dictionaries to keep track of counts of each label pattern
    lengths_seq_1 = dict()
    lengths_seq_2 = dict()
    lengths_seq_3 = dict()
    lengths_seq_4 = dict()
    
    frequen_seq_1 = dict()
    frequen_seq_2 = dict()
    frequen_seq_3 = dict()
    frequen_seq_4 = dict()
    
    medians_seq_1 = dict()
    medians_seq_2 = dict()
    medians_seq_3 = dict()
    medians_seq_4 = dict()
    
    for each_seq in unique_seq_1:
        lengths_seq_1[each_seq] = 0
        frequen_seq_1[each_seq] = 0
        medians_seq_1[each_seq] = 0
    for each_seq in unique_seq_2:
        lengths_seq_2[each_seq] = 0
        frequen_seq_2[each_seq] = 0
        medians_seq_2[each_seq] = 0
    for each_seq in unique_seq_3:
        lengths_seq_3[each_seq] = 0
        frequen_seq_3[each_seq] = 0
        medians_seq_3[each_seq] = 0
    for each_seq in unique_seq_4:
        lengths_seq_4[each_seq] = 0
        frequen_seq_4[each_seq] = 0
        medians_seq_4[each_seq] = 0
    
    # prep plot counts against label_seq_1
    for each_seq in lengths_seq_1:
        indices = (df.label_seq_1 == each_seq)      # find indices of sequences
        lengths = df.orig_length[indices]           # grab the original tree sizes
        lengths_seq_1[each_seq] = np.mean(lengths)  # calculate the mean size
        medians_seq_1[each_seq] = np.median(lengths)# calculate median
        frequen_seq_1[each_seq] = np.sum(indices)   # get sequence total occurence
    
    # prep plot counts against label_seq_2
    for each_seq in lengths_seq_2:
        indices = (df.label_seq_2 == each_seq)      # find indices of sequences
        lengths = df.orig_length[indices]           # grab the original tree sizes
        lengths_seq_2[each_seq] = np.mean(lengths)  # calculate the mean size
        medians_seq_2[each_seq] = np.median(lengths)# calculate median
        frequen_seq_2[each_seq] = np.sum(indices)   # get sequence total occurence
        
    # prep plot counts against label_seq_3
    for each_seq in lengths_seq_3:
        indices = (df.label_seq_3 == each_seq)      # find indices of sequences
        lengths = df.orig_length[indices]           # grab the original tree sizes
        lengths_seq_3[each_seq] = np.mean(lengths)  # calculate the mean size
        medians_seq_3[each_seq] = np.median(lengths)# calculate median
        frequen_seq_3[each_seq] = np.sum(indices)   # get sequence total occurence
    
    # prep plot counts against label_seq_4
    for each_seq in lengths_seq_4:
        indices = (df.label_seq_4 == each_seq)      # find indices of sequences
        lengths = df.orig_length[indices]           # grab the original tree sizes
        lengths_seq_4[each_seq] = np.mean(lengths)  # calculate the mean size
        medians_seq_4[each_seq] = np.median(lengths)# calculate median
        frequen_seq_4[each_seq] = np.sum(indices)   # get sequence total occurence
        
    ''' Extract data into lists '''
    leng1_x = list(lengths_seq_1.keys())
    leng1_y = list(lengths_seq_1.values())
    median1x= list(medians_seq_1.keys())
    median1y= list(medians_seq_1.values())
    freq1_x = list(frequen_seq_1.keys())
    freq1_y = list(frequen_seq_1.values())
    
    leng2_x = list(lengths_seq_2.keys())
    leng2_y = list(lengths_seq_2.values())
    median2x= list(medians_seq_2.keys())
    median2y= list(medians_seq_2.values())
    freq2_x = list(frequen_seq_2.keys())
    freq2_y = list(frequen_seq_2.values())
    
    leng3_x = list(lengths_seq_3.keys())
    leng3_y = list(lengths_seq_3.values())
    median3x= list(medians_seq_3.keys())
    median3y= list(medians_seq_3.values())
    freq3_x = list(frequen_seq_3.keys())
    freq3_y = list(frequen_seq_3.values())
    
    leng4_x = list(lengths_seq_4.keys())
    leng4_y = list(lengths_seq_4.values())
    median4x= list(medians_seq_4.keys())
    median4y= list(medians_seq_4.values())
    freq4_x = list(frequen_seq_4.keys())
    freq4_y = list(frequen_seq_4.values())
    
    ''' Sort the lists ''' 
    sort_leng1 = sorted(zip(leng1_y, leng1_x), reverse=True)
    sort_medi1 = sorted(zip(median1y, median1x), reverse=True)
    sort_freq1 = sorted(zip(freq1_y, freq1_x), reverse=True)
    
    sort_leng2 = sorted(zip(leng2_y, leng2_x), reverse=True)
    sort_medi2 = sorted(zip(median2y, median2x), reverse=True)
    sort_freq2 = sorted(zip(freq2_y, freq2_x), reverse=True)
    
    sort_leng3 = sorted(zip(leng3_y, leng3_x), reverse=True)
    sort_medi3 = sorted(zip(median3y, median3x), reverse=True)
    sort_freq3 = sorted(zip(freq3_y, freq3_x), reverse=True)
    
    sort_leng4 = sorted(zip(leng4_y, leng4_x), reverse=True)
    sort_medi4 = sorted(zip(median4y, median4x), reverse=True)
    sort_freq4 = sorted(zip(freq4_y, freq4_x), reverse=True)
    
    ''' Extract the sorted lists '''
    sort_leng1_y, sort_leng1_x = zip(*sort_leng1)
    sort_median1y,sort_median1x= zip(*sort_medi1)
    sort_freq1_y, sort_freq1_x = zip(*sort_freq1)
    
    sort_leng2_y, sort_leng2_x = zip(*sort_leng2)
    sort_median2y,sort_median2x= zip(*sort_medi2)
    sort_freq2_y, sort_freq2_x = zip(*sort_freq2)
    
    sort_leng3_y, sort_leng3_x = zip(*sort_leng3)
    sort_median3y,sort_median3x= zip(*sort_medi3)
    sort_freq3_y, sort_freq3_x = zip(*sort_freq3)
    
    sort_leng4_y, sort_leng4_x = zip(*sort_leng4)
    sort_median4y,sort_median4x= zip(*sort_medi4)
    sort_freq4_y, sort_freq4_x = zip(*sort_freq4)
    
    seq2_freq_counts = []
    seq3_freq_counts = []
    seq4_freq_counts = []
    # TODO reached here
    """ Grab the most frequent sequences' averages """
    for each_pattern in sort_freq2_x[0:10]:
        seq2_freq_counts.append()
    
    for each_pattern in sort_freq3_x[0:10]:
        seq3_freq_counts.append()
    
    for each_pattern in sort_freq4_x[0:10]:
        seq4_freq_counts.append()
    
    ''' Set up the plots '''
    fig1, axes1 = plt.subplots(1,3)
    fig2, axes2 = plt.subplots(1,3)
    fig3, axes3 = plt.subplots(1,3)
    fig4, axes4 = plt.subplots(1,3)
    
    fig5, axes5 = plt.subplots(2,1)
    fig6, axes6 = plt.subplots(2,1)
    fig7, axes7 = plt.subplots(2,1)
    
    ax1_1 = axes1[0]
    ax1_2 = axes1[1]
    ax1_3 = axes1[2]
    
    ax2_1 = axes2[0]
    ax2_2 = axes2[1]
    ax2_3 = axes2[2]
    
    ax3_1 = axes3[0]
    ax3_2 = axes3[1]
    ax3_3 = axes3[2]
    
    ax4_1 = axes4[0]
    ax4_2 = axes4[1]
    ax4_3 = axes4[2]
    
    ax5_1 = axes5[0]
    ax5_2 = axes5[1]
    ax6_1 = axes6[0]
    ax6_2 = axes6[1]
    ax7_1 = axes7[0]
    ax7_2 = axes7[1]
    
    for each_ax in [ax1_1, ax1_2, ax1_3, ax2_1, ax2_2, ax2_3, 
                    ax3_2, ax3_3, ax4_2, ax4_3,
                    ax5_1, ax5_2, ax6_1, ax6_2, ax7_1, ax7_2]:
        each_ax.grid(True)
        plt.setp(each_ax.get_xticklabels(), rotation=90, ha="right")
    
    ax1_1.set_title('1-seqs avg tree size')
    ax1_1.bar(sort_leng1_x, sort_leng1_y)
    ax1_2.set_title('1-seqs median tree size')
    ax1_2.bar(sort_median1x, sort_median1y)
    ax1_3.set_title('1-seqs frequencies')
    ax1_3.bar(sort_freq1_x, sort_freq1_y)
    
    #ax2_1.set_title('2-seqs top/bottom 10 avg tree size')
    ax2_1.set_title('2-seqs avg tree size')
    ax2_1.bar(sort_leng2_x[0:10], sort_leng2_y[0:10])
    ax2_1.bar(sort_leng2_x[10:-10], sort_leng2_y[10:-10],color='orange')
    ax2_1.bar(sort_leng2_x[-10:], sort_leng2_y[-10:], color='red')
    limit2 = ax2_1.get_ylim()
    ax2_1.get_xaxis().set_ticks([])
    #ax2_1.bar(sort_leng2_x[0:10], sort_leng2_y[0:10])
    #ax2_1.bar(sort_leng2_x[-10:], sort_leng2_y[-10:], color='red')
    ax2_2.set_title('2-seqs top/bottom 10 avg tree size')
    ax2_2.bar(sort_leng2_x[0:10], sort_leng2_y[0:10])
    ax2_2.bar(sort_leng2_x[-10:], sort_leng2_y[-10:], color='red')
    ax2_3.set_title('2-seqs top 10 occurring')
    ax2_3.bar(sort_freq2_x[0:10], sort_freq2_y[0:10])
    
    #ax3_1.set_title('3-seqs top/bottom 10 avg tree size')
    ax3_1.set_title('3-seqs avg tree size')
    ax3_1.stem(sort_leng3_x[0:10], sort_leng3_y[0:10], markerfmt=',')
    ax3_1.stem(sort_leng3_x[10:-10], sort_leng3_y[10:-10], linefmt='orange', markerfmt=',')
    ax3_1.stem(sort_leng3_x[-10:], sort_leng3_y[-10:], linefmt='red', markerfmt=',')
    limit3 = ax3_1.get_ylim()
    ax3_1.get_xaxis().set_ticks([])
    #ax3_1.bar(sort_leng3_x[0:10], sort_leng3_y[0:10])
    #ax3_1.bar(sort_leng3_x[-10:], sort_leng3_y[-10:], color='red')
    ax3_2.set_title('3-seqs top/bottom 10 avg tree size')
    ax3_2.bar(sort_leng3_x[0:10], sort_leng3_y[0:10])
    ax3_2.bar(sort_leng3_x[-10:], sort_leng3_y[-10:], color='red')
    ax3_3.set_title('3-seqs top 10 occurring')
    ax3_3.bar(sort_freq3_x[0:10], sort_freq3_y[0:10])
    
    #ax4_1.set_title('4-seqs top/bottom 10 avg tree size')
    ax4_1.set_title('4-seqs avg tree size')
    ax4_1.stem(sort_leng4_x[0:10], sort_leng4_y[0:10], markerfmt=',')
    ax4_1.stem(sort_leng4_x[10:-10], sort_leng4_y[10:-10], linefmt='orange', markerfmt=',')
    ax4_1.stem(sort_leng4_x[-10:], sort_leng4_y[-10:], linefmt='red', markerfmt=',')
    limit4 = ax4_1.get_ylim()
    ax4_1.get_xaxis().set_ticks([])
    #ax4_1.bar(sort_leng4_x[0:10], sort_leng4_y[0:10])
    #ax4_1.bar(sort_leng4_x[-10:], sort_leng4_y[-10:], color='red')
    ax4_2.set_title('4-seqs top/bottom 10 avg tree size')
    ax4_2.bar(sort_leng4_x[0:10], sort_leng4_y[0:10])
    ax4_2.bar(sort_leng4_x[-10:], sort_leng4_y[-10:], color='red')
    ax4_3.set_title('4-seqs top 10 occurring')
    ax4_3.bar(sort_freq4_x[0:10], sort_freq4_y[0:10])
    
    ax5_1.set_title('2-seqs top 10 occurring')
    
    
    ax6_1.set_title('3-seqs top 10 occurring')
    
    ax7_1.set_title('4-seqs top 10 occurring')
    
    figures = plt.get_fignums()
    for each_figure in figures:
        plt.figure(each_figure)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
    return [lengths_seq_1, lengths_seq_2, lengths_seq_3, lengths_seq_4], [frequen_seq_1,frequen_seq_2,frequen_seq_3,frequen_seq_4], df
    # return [tree_sizes, post_labels], [seq_names, seq_counts, avg_lengths, med_lengths]

def tighten_plots():
    figures = plt.get_fignums()
    for each_figure in figures:
        plt.figure(each_figure)
        plt.tight_layout()

if __name__ =='__main__':
    '''
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
    length_seq, freq_seq, df = inspect_labels_vs_length()
    #train_df = reprocess_df()
    tighten_plots()
    
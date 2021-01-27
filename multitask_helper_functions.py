#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:56:32 2020

@author: jakeyap
"""
import torch
from sklearn.metrics import precision_recall_fscore_support as f1_help
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import logging

logger = logging.getLogger('__main__')


def rescale_labels(labels):
    '''
    Given a set of stance labels, shift all of them up by 1.
    
    The classes stored in the pkl and tsv files are 
    -1= no post    0 = deny
    1 = support    2 = query
    3 = comment
    
    When the labels enter the functions in this file,
    the -1 labels will mess up some of the functions 
    Added 1 to all of them to take care of this bug
    0 = no post    1 = deny
    2 = support    3 = query
    4 = comment

    Parameters
    ----------
    labels: tensor
        predicted stance labels from -1 to 3 is no post.
    
    Returns
    -------
    labels + 1.
    '''
    return labels + 1
    

def length_loss(pred_logits, true_labels, loss_fn, divide=9):
    '''
    Given a set of length labels and logits, calculate the binary cross entropy loss

    Parameters
    ----------
    predicted_logits: tensor
        raw logits from with dimensions [n, 2] where
            n: minibatch size
            2: 2 logits because we frame length prediction as binary classification
    
    true_labels : tensor
        original thread lengths with dimensions [n, A] where
            n: minibatch size
            A: number of posts in original thread
        
    loss_fn: loss function
        Takes in 2 tensors and calculates cross entropy loss
    
    divide : integer, default is 9. 
        Number to split thread lengths into a binary classification problem
        Number <= divide is class 0, >divide is class 1
    Returns
    -------
    loss, a scalar or vector. depending on loss's setting. 
    '''
    binary_labels = (true_labels>=divide)*1 # convert to binary labels. shape=(n,1)
    binary_labels = binary_labels.view(-1)  # cast it into shape=(N,)
    binary_labels = binary_labels.long()    # convert into long datatype
    return loss_fn(pred_logits, binary_labels)

def length_loss_4class(pred_logits, true_labels, loss_fn, divide=[5,9,15]):
    '''
    Given a set of length labels and logits, calculate the binary cross entropy loss

    Parameters
    ----------
    predicted_logits: tensor
        raw logits from with dimensions [n, A] where
            n: minibatch size
            A: num of classes in length. 4
    
    true_labels : tensor
        original thread lengths with dimensions [n, A] where
            n: minibatch size
            A: number of posts in original thread
        
    loss_fn: loss function
        Takes in 2 tensors and calculates cross entropy loss
    
    divide : integer, default is [5,9,15]. 
        Numbers to split thread lengths into 
    Returns
    -------
    loss, a scalar or vector. depending on loss's setting. 
    '''
    
    labels = 0 * (true_labels < divide[0])      # split into buckets
    labels += 1 * ((true_labels >= divide[1]) & 
                   (true_labels < divide[2]))
    labels += 2 * ((true_labels >= divide[2]) & 
                   (true_labels < divide[3]))
    labels += 3 * (true_labels >= divide[3])
    
    labels = labels.view(-1)    # cast it into shape=(N,)
    labels = labels.long()      # convert into long datatype
    
    return loss_fn(pred_logits, labels)

def stance_loss(pred_logits, true_labels, loss_fn):
    '''
    Given a set of class labels and logits, calculate the cross entropy loss

    Parameters
    ----------
    pred_logits: logit tensor with dimensions [n, A, B] where
        n: minibatch size
        A: number of posts in 1 thread
        B: number of classes
    
    true_labels : tensor with dimensions [n,1]
        actual stance labels where
            -1= no post    0 = deny
            1 = support    2 = query
            3 = comment
        
    loss_fn: loss function
        Takes in 2 tensors and calculates cross entropy loss
        
    Returns
    -------
    loss, a scalar or vector. depending on loss's setting. 
    '''
    num_labels = pred_logits.shape[-1]
    # cast it into shape=(NxA,C) tensor where N=minibatch, C=num of classes
    logits = pred_logits.view(-1, num_labels) 
    
    labels = true_labels.view(-1)       # cast it into shape=(N,)
    labels = labels.long()              # convert into long datatype
    new_labels = rescale_labels(labels) # rescale labels to take care of -1s
    #loss_fn.ignore_index = 0            # set the loss function to ignore 0 labels
    
    return loss_fn(logits, new_labels)

def logit_2_class_length(pred_logits):
    '''
    Converts a logit tensor into a label tensor
    
    Parameters
    ----------
    pred_logits: tensor
        raw logits from with dimensions [n, X] where
            n: minibatch size
            X: num classes in length prediction. 2 if binary, could be more classes
    Returns
    -------
    labels : tensor
        labels with dimensions [n,] where each element is 
            0 if it is short in length
            1 if it is long in length
    '''
    pred_labels = torch.argmax(pred_logits, axis=1)
    pred_labels = pred_labels.long()
    return pred_labels.reshape(-1)

def logit_2_class_stance(pred_logits):
    '''
    Converts a logit tensor into a label tensor
    
    Parameters
    ----------
    pred_logits: tensor
        raw logits from with dimensions [n, A, B] where
            n: minibatch size
            A: number of posts in 1 thread
            B: number of classes
    Returns
    -------
    labels : tensor
        labels with dimensions [n x A] where each element below
            0 = no post    1 = deny
            2 = support    3 = query
            4 = comment
    '''
    pred_labels = torch.argmax(pred_logits, axis=2)
    pred_labels = pred_labels.long()
    return pred_labels.reshape(-1)

def accuracy_length(pred_labels, true_labels, divide=9):
    '''
    Returns the accuracy of stance predictions 

    Parameters
    ----------
    pred_labels : 1D tensor, shape=(N,)
        predicted labels.
    true_labels : 1D tensor, shape=(N,)
        ground truth labels.
    divide : integer, default is 9
        Number to split thread lengths into a binary classification problem
        Number <= divide is class 0, >divide is class 1

    Returns
    -------
    accuracy : float
        [True Pos + True Neg] / [Total].
    '''
    # convert lengths to binary labels
    binarylabels = (true_labels >= divide) * 1
    count = true_labels.shape[0]
    num_correct = torch.sum(pred_labels == binarylabels).item()
    return num_correct / count

def accuracy_stance(pred_labels, true_labels, incl_empty=True):
    '''
    Returns the accuracy of stance predictions 

    Parameters
    ----------
    pred_labels : 1D tensor, shape=(N,)
        predicted labels.
    true_labels : 1D tensor, shape=(N,)
        ground truth labels.
    incl_empty : boolean. Default is True
        if True, count all the empty posts as part of accuracy
    Returns
    -------
    accuracy : float
        [True Pos + True Neg] / [Total].
    '''
    if incl_empty:
        count = true_labels.shape[0]
        # rescale labels to take care of -1s
        new_labels = rescale_labels(true_labels) 
        num_correct = torch.sum(pred_labels == new_labels).item()
        return num_correct / count
    else:
        # rescale labels to take care of -1s
        new_labels = rescale_labels(true_labels)
        sel_indices = (new_labels!=0)
        filt_new_labels = new_labels[sel_indices]
        filt_pred_labels= pred_labels[sel_indices]
        
        count = filt_new_labels.shape[0]
        num_correct = torch.sum(filt_new_labels == filt_pred_labels).item()
        return num_correct / count

def length_f1(pred_lengths, true_lengths, divide=9):
    '''
    Calculates the F1 score of the length prediction task

    Parameters
    ----------
    pred_lengths : tensor
        labels with dimensions [n,] where each element is 
            0 if it is short in length
            1 if it is long in length
    true_lengths : tensor
        labels with dimensions [n,] where each element is 
            0 if it is short in length
            1 if it is long in length
    divide : integer, default is 9
        Number to split thread lengths into a binary classification problem
        Number <= divide is class 0, >divide is class 1
    
    Returns
    -------
    precisions : tuple of floats, size=2
        Self explanatory
    recalls : tuple of floats, size=2
        Self explanatory
    f1scores : tuple of floats, size=2
        Self explanatory
    supports : tuple of floats, size=2
        How many samples
    accuracy : float
    '''
    binarylabels = (true_lengths >= divide) * 1                     # convert lengths to binary labels
    precisions, recalls, f1scores, supports = f1_help(binarylabels, # calculate loss based on binary labels
                                                      pred_lengths,
                                                      average=None,
                                                      labels=[0,1])
    f1_score_macro = sum(f1scores) / len(f1scores)
    return precisions, recalls, f1scores, supports, f1_score_macro

def length_f1_msg(precisions, recalls, f1scores, supports, f1_scores_macro):
    '''
    For printing the f1 score for length prediction task

    Parameters
    ----------
    precisions : tuple of length 2
    recalls : tuple of length 2
    f1scores : tuple of length 2
    supports : tuple of length 2
    f1_scores_macro : tuple of length 2

    Returns
    -------
    string : string
        for printing the f1 score nicely.

    '''
    string = 'Labels \t\tPrecision\tRecall\t\tF1 score\tSupport\n'
    string +='Short  \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[0],recalls[0],f1scores[0],supports[0])
    string +='Long   \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[1],recalls[1],f1scores[1],supports[1])
    string +='\n'
    string +='F1-macro\t%1.4f' % f1_scores_macro
    return string

def stance_f1(pred_stance, true_stance, incl_empty=True, coarse_disc=False):
    '''
    Calculates the f1 scores for each label category

    Parameters
    ----------
    pred_lengths : 1D tensor with shape (N,)
        binary predicted lengths. 0=short, 1=long
    true_lengths : 1D tensor with shape (N,)
        binary true lengths. 0=short, 1=long
    incl_empty : boolean. Default is true
        if true, include isEmpty label in calculations
    coarse_disc : boolean. Default is false
        if true, do the full 10 classes of coarse discourse dataset + isEmpty
        if false, do the 4 classes of SRQ dataset + isEmpty
    Returns
    -------
    precisions : tuple of floats, length=5
        precision score broken down by category
    recalls : tuple of floats, length=5
        recall score broken down by category
    f1scores : tuple of floats, length=5
        f1 score broken down by category
    supports : tuple of int, length=5
        number of actual samples by category
    f1_score_macro : float
        Macro averaged F1 score
    '''
    # remember to rescale the stance labels to take care of -1s
    new_stance = rescale_labels(true_stance)
    if coarse_disc:
        stance_labels = [1,2,3,4,5,6,7,8,9,10]
    else:
        stance_labels = [1,2,3,4]
    if incl_empty:
        stance_labels.insert(0,0)
        
    precisions, recalls, f1scores, supports = f1_help(new_stance, 
                                                  pred_stance, 
                                                  average=None,
                                                  labels=stance_labels)
    
    f1_score_macro = sum(f1scores) / len(f1scores)
    return precisions, recalls, f1scores, supports, f1_score_macro

def stance_f1_msg(precisions, recalls, f1scores, supports, f1_scores_macro, 
                  incl_empty=True, coarse_disc=False):
    '''
    For printing the f1 score for stance prediction task

    Parameters
    ----------
    precisions : tuple of length 5 or 11
    recalls : tuple of length 5 or 11
    f1scores : tuple of length 5 or 11
    supports : tuple of length 5 or 11
    f1_scores_macro : macro f1 score
    incl_empty : boolean to decide whether to include isempty label. default is True
    coarse_disc : boolean. Default is false
        if true, do the full 10 classes of coarse discourse dataset + isEmpty
        if false, do the 4 classes of SRQ dataset + isEmpty
    Returns
    -------
    string : string
        for printing the f1 score nicely.

    '''
    
    if coarse_disc:
        if incl_empty:
            string = 'Labels      \t\tPrecision\tRecall\t\tF1 score\tSupport\n'
            string +='Empty       \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[0],recalls[0],f1scores[0],supports[0])
            string +='Question    \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[1],recalls[1],f1scores[1],supports[1])
            string +='Answer      \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[2],recalls[2],f1scores[2],supports[2])
            string +='Announcement\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[3],recalls[3],f1scores[3],supports[3])
            string +='Agreement   \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[4],recalls[4],f1scores[4],supports[4])
            string +='Appreciation\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[5],recalls[5],f1scores[5],supports[5])
            string +='Disagreement\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[6],recalls[6],f1scores[6],supports[6])
            string +='-ve reaction\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[7],recalls[7],f1scores[7],supports[7])
            string +='Elaboration \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[8],recalls[8],f1scores[8],supports[8])
            string +='Humor       \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[9],recalls[9],f1scores[9],supports[9])
            string +='Other       \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[10],recalls[10],f1scores[10],supports[10])
            
            string +='\n'
            string +='F1-macro\t%1.4f\n' % f1_scores_macro
            string +='Excl empty\t%1.4f' % np.average(f1scores[1:])
        else:
            string = 'Labels      \t\tPrecision\tRecall\t\tF1 score\tSupport\n'
            string +='Question    \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[0],recalls[0],f1scores[0],supports[0])
            string +='Answer      \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[1],recalls[1],f1scores[1],supports[1])
            string +='Announcement\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[2],recalls[2],f1scores[2],supports[2])
            string +='Agreement   \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[3],recalls[3],f1scores[3],supports[3])
            string +='Appreciation\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[4],recalls[4],f1scores[4],supports[4])
            string +='Disagreement\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[5],recalls[5],f1scores[5],supports[5])
            string +='-ve reaction\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[6],recalls[6],f1scores[6],supports[6])
            string +='Elaboration \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[7],recalls[7],f1scores[7],supports[7])
            string +='Humor       \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[8],recalls[8],f1scores[8],supports[8])
            string +='Other       \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[9],recalls[9],f1scores[9],supports[9])
            
            string +='\n'
            string +='F1-macro\t%1.4f' % f1_scores_macro
    else:
        if incl_empty:
            string = 'Labels \t\tPrecision\tRecall\t\tF1 score\tSupport\n'
            string +='Empty  \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[0],recalls[0],f1scores[0],supports[0])
            string +='Deny   \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[1],recalls[1],f1scores[1],supports[1])
            string +='Support\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[2],recalls[2],f1scores[2],supports[2])
            string +='Query  \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[3],recalls[3],f1scores[3],supports[3])
            string +='Comment\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[4],recalls[4],f1scores[4],supports[4])
            string +='\n'
            string +='F1-macro\t%1.4f\n' % f1_scores_macro
            string +='Excl empty\t%1.4f' % np.average(f1scores[1:])
        else:
            string = 'Labels \t\tPrecision\tRecall\t\tF1 score\tSupport\n'
            string +='Deny   \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[0],recalls[0],f1scores[0],supports[0])
            string +='Support\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[1],recalls[1],f1scores[1],supports[1])
            string +='Query  \t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[2],recalls[2],f1scores[2],supports[2])
            string +='Comment\t\t%1.4f    \t%1.4f \t\t%1.4f   \t%d\n' % (precisions[3],recalls[3],f1scores[3],supports[3])
            string +='\n'
            string +='F1-macro\t%1.4f' % f1_scores_macro
    return string

def plot_confusion_matrix(y_true, y_pred, labels, label_names):
    '''
    Generates confusion matrix and plots it.
    Parameters
    ----------
    y_true : linear numpy array of true labels
    y_pred : linear numpy array of predicted labels
    label_ticks : list of ints for axes ticks in confusion matrix
    label_names : list of strings. For labelling axes
        DESCRIPTION.

    Returns
    -------
    matrix0 : 2d numpy array. confusion matrix.
    '''
    
    fig0, ax0 = plt.subplots()
    # calculate confusion matrix
    matrix0 = confusion_matrix(y_true, y_pred, labels)
    
    label_list_len = len(labels)
        
    plt.imshow(matrix0, cmap='gray')
    
    # We want to show all ticks...
    ax0.set_xticks(range(label_list_len))
    ax0.set_yticks(range(label_list_len))
    
    # ... and label them with the respective list entries
    ax0.set_xticklabels(label_names, size=10)
    ax0.set_yticklabels(label_names, size=10)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax0.get_xticklabels(), rotation=45, 
             ha="right",rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    
    maxvalue = matrix0.max()                    # find max value of conf matrix
    minvalue = matrix0.min()                    # find min value of conf matrix
    midpoint = (maxvalue + minvalue) / 2        # find mid point of conf matrix
    for i in range(label_list_len):             # loop through all rows in matrix
        for j in range(label_list_len):         # loop through all cols in matrix
            number = matrix0[i, j]              # get count for labelling each box
            if number > midpoint:
                ax0.text(j, i, str(number),     # for large nums, write in black
                         ha="center", va="center", 
                         color="black", size=10)
            else:
                ax0.text(j, i, str(number),     # for small nums, write in white
                         ha="center", va="center", 
                         color="white", size=10)
    
    plt.colorbar(aspect=50)                 # draw color bar
    plt.title('Confusion matrix', size=15)  # title
    plt.ylabel('True Label', size=12)       # ylabel
    plt.xlabel('Predicted label', size=12)  # xlabel
    plt.tight_layout()
    return matrix0
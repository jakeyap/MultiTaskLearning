#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:56:32 2020

@author: jakeyap
"""
import torch
from sklearn.metrics import precision_recall_fscore_support as f1_help

def rescale_labels(labels):
    '''
    Given a set of stance labels, shift all of them up by 1.
    
    The classes stored in the pkl and tsv files are 
    -1= no post    0 = deny
    1 = support    2 = query
    3 = comment
    
    When the labels enter the functions in this, calculator
    the -1 will screw up some of the library functions 
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
    Given a set of length labels, calculate the binary cross entropy loss

    Parameters
    ----------
    predicted_logits: tensor
        raw logits from with dimensions [n, 2] where
            n: minibatch size
            2: 2 logits because we frame length prediction as binary classification
    
    true_labels : tensor
        original thread lengths with dimenstions [n, A] where
            n: minibatch size
            A: number of posts in original thread
        
    loss_fn: loss function
        Takes in 2 tensors and calculates cross entropy loss
    
    divide : integer, default is 9
        Number to split thread lengths into a binary classification problem
        Number <= divide is class 0, >divide is class 1
    Returns
    -------
    loss, a scalar or vector. depending on loss's setting. 
    '''
    
    binary_labels = (true_labels >= divide) # convert to binary labels. shape=(n,1)
    binary_labels = binary_labels.view(-1)  # cast it into shape=(N,)
    binary_labels = binary_labels.long()    # convert into long datatype
    
    return loss_fn(pred_logits, binary_labels)

def stance_loss(pred_logits, true_labels, loss_fn):
    '''
    Given a set of class labels, calculate the cross entropy loss

    Parameters
    ----------
    pred_logits: tensor
        raw logits from with dimensions [n, A, B] where
            n: minibatch size
            A: number of posts in 1 thread
            B: number of classes
    
    true_labels : tensor
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
    # cast it into shape=(N,C) tensor where N=minibatch, C=num of classes
    logits = pred_logits.view(-1, num_labels) 
    
    labels = true_labels.view(-1)       # cast it into shape=(N,)
    labels = labels.long()              # convert into long datatype
    new_labels = rescale_labels(labels) # rescale labels to take care of -1s
    loss_fn.ignore_index = 0            # set the loss function to ignore 0 labels
    return loss_fn(logits, new_labels)

def logit_2_class_length(pred_logits):
    '''
    Converts a logit tensor into a label tensor
    
    Parameters
    ----------
    pred_logits: tensor
        raw logits from with dimensions [n, 2] where
            n: minibatch size
            2: number of classes in binary classification
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

def accuracy(pred_labels, labels):
    return torch.sum(pred_labels == labels)

def macro_f1(pred_labels, y_true):
    precision, recall, f1score, support = f1_help(y_true, 
                                                  pred_labels, 
                                                  average='macro')
    return precision, recall, f1score, support 
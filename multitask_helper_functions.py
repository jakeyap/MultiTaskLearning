#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:56:32 2020

@author: jakeyap
"""

def stance_loss(predicted_labels, actual_labels, loss_fn):
    '''
    Given a set of stance labels, calculate the proper loss while accounting 
    for positions labelled -1. 
    
    Just add 1 to all the predicted labels and actual labels and calculate 
    anyway. 
    The old class numbers are 
    -1= no post    0 = deny
    1 = support    2 = query
    3 = comment
    
    Add 1 to all to give this
    0 = no post    1 = deny
    2 = support    3 = query
    4 = comment

    Parameters
    ----------
    predicted_labels: tensor
        predicted stance labels from -1 to 3 is no post.
    
    actual_labels : tensor
        actual stance labels where -1 to 3 is no post.
        
    loss_fn: loss function
        Takes in 2 tensor and calculates cross entropy loss
    Returns
    -------
    loss.

    '''
    
    return loss_fn(predicted_labels + 1, actual_labels + 1)
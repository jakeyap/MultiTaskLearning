# -*- coding: utf-8 -*-

"""
Created on Mon Sep 14 2020 11:19 

@author: jakeyap
"""
import csv

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
        return lines
    

# -*- coding: utf-8 -*-

"""
Created on Mon Sep 14 2020 11:19 

@author: jakeyap
"""

from transformers import BertTokenizer
import csv
default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
default_tokenizer.add_tokens(['[deleted]', '[URL]','[empty]'])

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, lineid, orig_length, text, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.lineid = lineid
        self.text = text
        self.orig_length = orig_length
        self.labels = labels

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
    counter = 0
    examples = []
    # remember to skip headers
    for eachline in tsv_lines[1:]:
        lineid = counter
        labels_list = eachline[1].split(',')
        orig_length = int(eachline[2])
        text = eachline[3].lower().split('|||||')
        examples.append(
            InputExample(lineid=lineid, orig_length=orig_length, 
                         text=text, labels=labels_list))
        counter = counter + 1
    return examples

def get_test_set_shuffled(filename='./data/combined/shuffled_test.tsv'):
    return open_tsv_data(filename)

def get_dev_set_shuffled(filename='./data/combined/shuffled_dev.tsv'):
    return open_tsv_data(filename)

def get_train_set_shuffled(filename='./data/combined/shuffled_train.tsv'):
    return open_tsv_data(filename)

def get_test_set(filename='./data/combined/combined_test.tsv'):
    return open_tsv_data(filename)

def get_dev_set(filename='./data/combined/combined_dev.tsv'):
    return open_tsv_data(filename)

def get_train_set(filename='./data/combined/combined_train.tsv'):
    return open_tsv_data(filename)
        
def tokenize():
    return
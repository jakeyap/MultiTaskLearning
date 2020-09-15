#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:32:49 2020

@author: jakeyap
"""
import torch
import torch.optim as optim
import transformers

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import DataProcessor
from classifier_models import my_ModelA0

from sklearn.metrics import precision_recall_fscore_support

'''======== FILE NAMES FOR LOGGING ========'''
FROM_SCRATCH = True
'''======== HYPERPARAMETERS START ========'''
NUM_TO_PROCESS = 100000
BATCH_SIZE_TRAIN = 40
BATCH_SIZE_TEST = 40
LOG_INTERVAL = 10

N_EPOCHS = 80
LEARNING_RATE = 0.001
MOMENTUM = 0.5

MAX_POST_LENGTH = 256
MAX_POST_PER_THREAD = 4

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''
directory = './data/combined/'
filename = 'encoded_combined_test.pkl'

gpu = torch.device("cuda")
n_gpu = torch.cuda.device_count()
if FROM_SCRATCH:
    #config = BertConfig.from_pretrained('bert-base-uncased')
    #config.num_labels = 2
    model = my_ModelA0.from_pretrained('bert-base-uncased',
                                       stance_num_labels=4,
                                       length_num_labels=2,
                                       max_post_num=MAX_POST_PER_THREAD, 
                                       max_post_length=MAX_POST_LENGTH)
    # Resize model vocab
    model.resize_token_embeddings(len(DataProcessor.default_tokenizer))
    
    # Move model into GPU
    model.to(gpu)
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)

# Load some data
dataframe = DataProcessor.load_df_from_pkl(directory+filename)
dataloader = DataProcessor.dataframe_2_dataloader(dataframe,batchsize=4,randomize=False,DEBUG=False)
# Feed into the model and see if it is working correctly

# batch_idx, minibatch = next(enumerate(dataloader))
for batch_idx, minibatch in enumerate(dataloader):
    encoded_comments = minibatch[1]
    token_type_ids = minibatch[1]
    attention_masks = minibatch[1]
    orig_length = minibatch[1]
    stance_labels = minibatch[1]
    dataset = TensorDataset(posts_index,
                        encoded_comments,
                        token_type_ids,
                        attention_masks,
                        orig_length,
                        stance_labels)
    
'''
'''
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
import logging, sys
import datetime
import time

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


timestamp = time.time()
timestamp = str(timestamp)

# for storing logs into a file
file_handler = logging.FileHandler(filename='./log_files/'+timestamp+'_log.log')
# for printing onto terminal
stdout_handler = logging.StreamHandler(sys.stdout)

# for storing training / dev / test losses
lossfile = open('./log_files/'+timestamp+'_losses.log','w')

handlers = [file_handler, stdout_handler]
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    handlers=handlers,
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Define the optimizer. Use SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)

# Load some data
dataframe = DataProcessor.load_df_from_pkl(directory+filename)

dataloader = DataProcessor.dataframe_2_dataloader(dataframe,
                                                  batchsize=4,
                                                  randomize=True,
                                                  DEBUG=False)
# Feed into the model and see if it is working correctly
'''
batch_idx, minibatch = next(enumerate(dataloader))
'''
counter = 0
for batch_idx, minibatch in enumerate(dataloader):
    posts_index = minibatch[0]
    encoded_comments = minibatch[1]
    token_type_ids = minibatch[2]
    attention_masks = minibatch[3]
    orig_length = minibatch[4]
    stance_labels = minibatch[5]
    #TODO reached here
    #posts_index = posts_index.to(gpu)
    stance_pred = model(input_ids = encoded_comments,
                  token_type_ids = token_type_ids, 
                  attention_masks = attention_masks, 
                  task='stance')
    '''
    '''
    logger.info('mini batch id %d', counter)
    logger.info(posts_index)
    counter = counter + 1
    break
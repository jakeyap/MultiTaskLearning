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
import multitask_helper_functions as helper
from classifier_models import my_ModelA0

from sklearn.metrics import precision_recall_fscore_support
import logging, sys
import datetime
import time
import matplotlib.pyplot as plt

'''======== FILE NAMES FOR LOGGING ========'''
FROM_SCRATCH = True
'''======== HYPERPARAMETERS START ========'''
NUM_TO_PROCESS = 100000
BATCH_SIZE_TRAIN = 3
BATCH_SIZE_TEST = 40
LOG_INTERVAL = 10

N_EPOCHS = 20
LEARNING_RATE = 0.0001
MOMENTUM = 0.25

MAX_POST_LENGTH = 256
MAX_POST_PER_THREAD = 4

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''
directory = './data/combined/'
test_filename = 'encoded_shuffled_test.pkl'
train_filename = 'encoded_shuffled_train.pkl'
dev_filename = 'encoded_shuffled_dev.pkl'


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
                                       stance_num_labels=5,
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
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ignore 0 labels - no post
# do averaging on the losses
loss_function = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)



# Load the test and training data
test_dataframe = DataProcessor.load_df_from_pkl(directory + test_filename)
dev_dataframe = DataProcessor.load_df_from_pkl(directory + dev_filename)
train_dataframe = DataProcessor.load_df_from_pkl(directory + train_filename)

train_dataloader = DataProcessor.dataframe_2_dataloader(train_dataframe,
                                                        batchsize=BATCH_SIZE_TRAIN,
                                                        randomize=True,
                                                        DEBUG=False)
test_dataloader = DataProcessor.dataframe_2_dataloader(test_dataframe,
                                                       batchsize=BATCH_SIZE_TRAIN,
                                                       randomize=False,
                                                       DEBUG=False)
dev_dataloader = DataProcessor.dataframe_2_dataloader(dev_dataframe,
                                                      batchsize=BATCH_SIZE_TRAIN,
                                                      randomize=False,
                                                      DEBUG=False)

# Feed into the model and see if it is working correctly
'''
batch_idx, minibatch = next(enumerate(dataloader))
'''

def train(epochs):
    # Set network into training mode to enable dropout
    model.train()
    
    losses = []
    counter = 0
    
    while counter < N_EPOCHS:
        logger.info('EPOCH: %d', counter)
        for batch_idx, minibatch in enumerate(train_dataloader):
            posts_index = minibatch[0]
            encoded_comments = minibatch[1]
            token_type_ids = minibatch[2]
            attention_masks = minibatch[3]
            orig_length = minibatch[4]
            stance_labels = minibatch[5]
            
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            orig_length = orig_length.to(gpu)
            stance_labels = stance_labels.to(gpu)
            
            stance_pred = model(input_ids = encoded_comments,
                                token_type_ids = token_type_ids, 
                                attention_masks = attention_masks, 
                                task='stance')
            stance_loss = helper.stance_loss(predicted_labels = stance_pred,
                                             actual_labels = stance_labels, 
                                             loss_fn = loss_function)
            stance_loss.backward()
            optimizer.step()
            loss = stance_loss.item()
            losses.append(loss)
            
            logger.info('loss: %1.3f', loss)
        counter = counter + 1

    return losses

if __name__ == '__main__':
    start_time = time.time()
    losses = train(N_EPOCHS)
    time_end = time.time()
    time_taken = time_end - start_time  
    print('Time elapsed: %6.2fs' % time_taken)
    plt.plot(losses)

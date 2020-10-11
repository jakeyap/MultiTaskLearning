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
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 40
LOG_INTERVAL = 10

N_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.25

MAX_POST_LENGTH = 256
MAX_POST_PER_THREAD = 4
THREAD_LENGTH_DIVIDER = 9 # for dividing thread lengths into binary buckets

PRINT_PICTURE = False
'''======== HYPERPARAMETERS END ========'''
directory = './data/combined/'
test_filename = 'encoded_shuffled_test_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
test_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
train_filename = 'encoded_shuffled_train_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
dev_filename = 'encoded_shuffled_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
#train_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)

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
torch.cuda.empty_cache()
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
    
    # Define the optimizers. Use SGD
    stance_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                 momentum=MOMENTUM)
    length_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                 momentum=MOMENTUM)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ignore 0 labels - no post
# do averaging on the losses
#stance_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
stance_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
length_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')



# Load the test and training data
print('Opening training data: ' + train_filename)
print('Opening dev data: ' + dev_filename)
print('Opening test data: ' + test_filename)

test_dataframe = DataProcessor.load_df_from_pkl(directory + test_filename)
dev_dataframe = DataProcessor.load_df_from_pkl(directory + dev_filename)
train_dataframe = DataProcessor.load_df_from_pkl(directory + train_filename)

print('Converting pickle data into dataloaders')
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
batch_idx, minibatch = next(enumerate(train_dataloader))
'''

def train(epochs):
    # Set network into training mode to enable dropout
    model.train()
    
    stance_losses = []  # for storing training losses of stance
    length_losses = []  # for storing training losses of length
    counter = 0
    
    while counter < N_EPOCHS:
        logger.info('EPOCH: %d', counter)
        for batch_idx, minibatch in enumerate(train_dataloader):
            # get the features from dataloader
            # posts_index = minibatch[0]
            encoded_comments = minibatch[1]
            token_type_ids = minibatch[2]
            attention_masks = minibatch[3]
            length_labels = minibatch[4]
            stance_labels = minibatch[5]
            
            # move features to gpu
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            length_labels = length_labels.to(gpu)
            stance_labels = stance_labels.to(gpu)
            
            # get the stance prediction logits
            stance_logits = model(input_ids = encoded_comments,
                                  token_type_ids = token_type_ids, 
                                  attention_masks = attention_masks, 
                                  task='stance')
            # calculate the stance loss
            stance_loss = helper.stance_loss(pred_logits = stance_logits,
                                             true_labels = stance_labels, 
                                             loss_fn = stance_loss_fn)
            
            stance_loss.backward()          # back propagation
            stance_optimizer.step()         # step the gradients once
            stance_optimizer.zero_grad()    # clear the gradients before the next step
            loss1 = stance_loss.item()      # get the value of the loss
            stance_losses.append(loss1)     # archive the loss
            
            # get the length prediction logits
            length_logits = model(input_ids = encoded_comments,
                                  token_type_ids = token_type_ids, 
                                  attention_masks = attention_masks, 
                                  task='length')
            # calculate the length loss
            length_loss = helper.length_loss(pred_logits = length_logits,
                                             true_labels = length_labels, 
                                             loss_fn = length_loss_fn,
                                             divide = THREAD_LENGTH_DIVIDER)
            length_loss.backward()          # back propagation
            length_optimizer.step()         # step the gradients once
            length_optimizer.zero_grad()    # clear the gradients before the next step
            loss2 = length_loss.item()      # get the value of the loss
            length_losses.append(loss2)     # archive the loss
            
            logger.info('\tstance loss: %1.3f\tlength loss: %1.3f' % (loss1, loss2))     # print losses
        counter = counter + 1

    return stance_losses, length_losses

def test(save=False):
    # Set network into evaluation mode
    model.eval()
    
    test_loss = 0
    num_correct = 0
    
    # start the label arrays. 1st data point has to be deleted later
    stance_pred_arr = torch.tensor([0],dtype=torch.int64)
    stance_true_arr = torch.tensor([0],dtype=torch.int64)
    length_pred_arr = torch.tensor([0],dtype=torch.int64)
    length_true_arr = torch.tensor([0],dtype=torch.int64)
    
    with torch.no_grad():
        for batch_idx, minibatch in enumerate(test_dataloader):
            # get the features from dataloader
            # posts_index = minibatch[0]
            encoded_comments = minibatch[1]
            token_type_ids = minibatch[2]
            attention_masks = minibatch[3]
            length_labels = minibatch[4]
            stance_labels = minibatch[5]
            
            # move features to gpu
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            length_labels = length_labels.to(gpu)
            stance_labels = stance_labels.to(gpu)
            
            stance_logits = model(input_ids = encoded_comments,         # get the stance prediction logits
                                  token_type_ids = token_type_ids,      # (n,A,B): n=minibatch, A=max_posts_per_thread, B=num of classes
                                  attention_masks = attention_masks, 
                                  task='stance')
            stance_pred = helper.logit_2_class_stance(stance_logits)    # convert logits to stance labels. (nA,)
            stance_pred_arr = torch.cat((stance_pred_arr,               # store all stance predictions in a linear array
                                         stance_pred.to('cpu')),0)
            stance_labels = stance_labels.reshape(stance_pred.shape)    # reshape from (n,A) into (nA,)
            stance_labels = stance_labels.long()                        # make sure the stance_labels datatype is int64
            stance_true_arr = torch.cat((stance_true_arr,               # store all stance labels in a linear array
                                         stance_labels.to('cpu')),0)
            
            length_logits = model(input_ids = encoded_comments,         # get the length prediction logits
                                  token_type_ids = token_type_ids,      # (n,2): n=minibatch, 2=num of classes
                                  attention_masks = attention_masks, 
                                  task='length')
            length_pred = helper.logit_2_class_length(length_logits)    # convert logits to length labels. (n,)
            length_pred_arr = torch.cat((stance_pred_arr,               # store all length predictions in a linear array
                                         stance_pred.to('cpu')),
                                        0)
            length_true_arr = torch.cat((stance_true_arr,               # store all length labels in a linear array
                                         stance_labels.to('cpu')),
                                        0)
            
    # remember to discard the 1st data point
    return stance_pred_arr[1:], stance_true_arr[1:], length_pred_arr[1:], length_true_arr[1:]

if __name__ == '__main__':
    start_time = time.time()
    losses_stance, losses_length = train(N_EPOCHS)
    pred_stance, true_stance, pred_length, true_length = test()
    
    time_end = time.time()
    time_taken = time_end - start_time  
    print('Time elapsed: %6.2fs' % time_taken)
    fig,axes = plt.subplots(2,1)
    ax1,ax2 = axes[0], axes[1]
    ax1.plot(losses_stance, color='red')
    ax2.plot(losses_length, color='blue')
    

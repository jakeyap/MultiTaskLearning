#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:32:49 2020

@author: jakeyap
"""
import torch
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import DataProcessor
import multitask_helper_functions as helper
from classifier_models import my_ModelA0

import logging, sys, argparse
import time
import matplotlib.pyplot as plt

# unused for now
def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after going through BERT tokenizer. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        help="Minibatch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Minibatch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--max_tweet_num', type=int, default=30, help="the maximum number of tweets")
    parser.add_argument('--max_tweet_length', type=int, default=17, help="the maximum length of each tweet")

    args = parser.parse_args()
    return args

'''======== FILE NAMES FOR LOGGING ========'''
FROM_SCRATCH = True
'''======== HYPERPARAMETERS START ========'''
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2
LOG_INTERVAL = 1

N_EPOCHS = 20
LEARNING_RATE = 0.001
MOMENTUM = 0.25

MAX_POST_LENGTH = 256
MAX_POST_PER_THREAD = 4
THREAD_LENGTH_DIVIDER = 9 # for dividing thread lengths into binary buckets

'''======== HYPERPARAMETERS END ========'''
directory = './data/combined/'
test_filename = 'encoded_shuffled_test_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
train_filename = 'encoded_shuffled_train_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
dev_filename = 'encoded_shuffled_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)


test_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
train_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
dev_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)

timestamp = time.time()
timestamp = str(timestamp)[5:]

model_savefile = './saved_models/'+timestamp+'_ModelA0.bin'

# for storing logs into a file
file_handler = logging.FileHandler(filename='./log_files/'+timestamp+'_log.log')
# for printing onto terminal
stdout_handler = logging.StreamHandler(sys.stdout)

# for storing training / dev / test losses
lossfile = './log_files/'+timestamp+'_losses.bin'
plotfile = './log_files/'+timestamp+'_plot.png'

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
    model = my_ModelA0.from_pretrained('bert-base-uncased',
                                       stance_num_labels=5,
                                       length_num_labels=2,
                                       max_post_num=MAX_POST_PER_THREAD, 
                                       max_post_length=MAX_POST_LENGTH)
    # Resize model vocab
    model.resize_token_embeddings(len(DataProcessor.default_tokenizer))
    
    # Move model into GPU
    model.to(gpu)
    logger.info('Running on %d GPUs' % n_gpu)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Define the optimizers. Use SGD
    stance_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                 momentum=MOMENTUM)
    length_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                 momentum=MOMENTUM)
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    logger.info('Load from saved model not implemented yet')
    
# ignore 0 labels - no post
# do averaging on the losses
#stance_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
stance_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
length_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

logger.info('\n===== Hyperparameters ======')
logger.info('BATCH_SIZE_TRAIN: %d' % BATCH_SIZE_TRAIN)
logger.info('LEARNING_RATE: %d' % LEARNING_RATE)
logger.info('MOMENTUM: %d' % MOMENTUM)
logger.info('MAX_POST_LENGTH: %d' % MAX_POST_LENGTH)
logger.info('MAX_POST_PER_THREAD: %d' % MAX_POST_PER_THREAD)
logger.info('THREAD_LENGTH_DIVIDER: %d' % THREAD_LENGTH_DIVIDER)

# Load the test and training data
logger.info('\n===== Loading data ======')
logger.info('Opening training data: ' + train_filename)
logger.info('Opening dev data: ' + dev_filename)
logger.info('Opening test data: ' + test_filename)
test_dataframe = DataProcessor.load_from_pkl(directory + test_filename)
dev_dataframe = DataProcessor.load_from_pkl(directory + dev_filename)
train_dataframe = DataProcessor.load_from_pkl(directory + train_filename)

logger.info('Converting pickle data into dataloaders')
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

def train(epochs):    
    best_f1_score = 0   # For tracking best model on dev set so far
    
    train_stance_losses = []    # For storing training losses of stance
    train_length_losses = []    # For storing training losses of length
    train_horz_index = []       # For horizontal axis during plotting
    
    dev_f1_scores = []          # For storing F1 scores of validation set
    dev_stance_losses = []      # For storing validation losses of stance
    dev_length_losses = []      # For storing validation losses of length
    dev_horz_index = []         # For horizontal axis during plotting
    
    for epoch_counter in range(epochs):
        model.train()       # Set network into training mode to enable dropout
        for batch_idx, minibatch in enumerate(train_dataloader):
            if batch_idx % LOG_INTERVAL == 0:
                logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch_counter, batch_idx)))
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
            
            stance_loss.backward()              # back propagation
            stance_optimizer.step()             # step the gradients once
            stance_optimizer.zero_grad()        # clear the gradients before the next step
            loss1 = stance_loss.item()          # get the value of the loss
            train_stance_losses.append(loss1)   # archive the loss
            
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
            length_loss.backward()              # back propagation
            length_optimizer.step()             # step the gradients once
            length_optimizer.zero_grad()        # clear the gradients before the next step
            loss2 = length_loss.item()          # get the value of the loss
            train_length_losses.append(loss2)   # archive the loss
            
            # Store the horizontal counter for minibatches seen
            train_horz_index.append(epoch_counter * len(train_dataloader) + batch_idx)
        
        # after epoch, run test on dev set
        test_results = test(mode='dev')
        stance_pred = test_results[0]
        stance_true = test_results[1]
        length_pred = test_results[2]
        length_true = test_results[3]
        dev_loss_stance = test_results[4]
        dev_loss_length = test_results[5]
        
        # store the dev set losses
        dev_stance_losses.append(dev_loss_stance)
        dev_length_losses.append(dev_loss_length)
        dev_horz_index.append(train_horz_index[-1])
        
        # calculate stance and length f1 metrics
        stance_metrics = helper.stance_f1(stance_pred, stance_true)
        length_metrics = helper.length_f1(length_pred, length_true, divide=THREAD_LENGTH_DIVIDER)
        
        stance_msg = helper.stance_f1_msg(stance_metrics[0],stance_metrics[1],stance_metrics[2],stance_metrics[3],stance_metrics[4])
        length_msg = helper.length_f1_msg(length_metrics[0],length_metrics[1],length_metrics[2],length_metrics[3],length_metrics[4])
        
        logger.info('\n' + stance_msg)
        logger.info('\n' + length_msg)
        
        f1_stance = stance_metrics[-1]      # extract macro f1 from tuple
        f1_length = length_metrics[-1]      # extract macro f1 from tuple
        f1_new = (f1_stance+f1_length) / 2  # average the 2 f1 scores
        dev_f1_scores.append(f1_new)
        
        if (f1_new > best_f1_score):
            best_f1_score = f1_new          # update the score
            save_model()                    # save the model
        
    return train_stance_losses, train_length_losses, train_horz_index, \
            dev_stance_losses, dev_length_losses, dev_f1_scores, dev_horz_index

def test(mode='test', save=False):
    '''
    mode must be 'test' or 'dev'.
    to select between test or validation modes
    '''
    # Set network into evaluation mode
    model.eval()
    
    # start the label arrays. 1st data point has to be deleted later
    stance_pred_arr = torch.tensor([0],dtype=torch.int64)
    stance_true_arr = torch.tensor([0],dtype=torch.int64)
    length_pred_arr = torch.tensor([0],dtype=torch.int64)
    length_true_arr = torch.tensor([0],dtype=torch.int64)
    
    if mode=='test':
        dataloader = test_dataloader
    elif mode=='dev':
        dataloader = dev_dataloader
    
    length_losses = []  # for tracking minibatch losses
    stance_losses = []  # for tracking minibatch losses
    
    with torch.no_grad():
        for batch_idx, minibatch in enumerate(dataloader):
            if batch_idx % LOG_INTERVAL == 0:
                logger.info(('\tTesting '+ mode +' set minibatch: %4d' % batch_idx))
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
            
            stance_loss = helper.stance_loss(pred_logits=stance_logits, # calculate the dev set stance loss
                                             true_labels=stance_labels, 
                                             loss_fn=stance_loss_fn)
            stance_losses.append(stance_loss)                           # store into list
            
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
            
            length_loss = helper.length_loss(pred_logits=length_logits, # calculate the dev set length loss
                                             true_labels=length_labels, 
                                             loss_fn=length_loss_fn,
                                             divide=THREAD_LENGTH_DIVIDER)
            length_losses.append(length_loss)                           # store into list
            
            length_pred = helper.logit_2_class_length(length_logits)    # convert logits to length labels. (n,)
            length_pred_arr = torch.cat((length_pred_arr,               # store all length predictions in a linear array
                                         length_pred.to('cpu')),
                                        0)
            length_labels = length_labels.reshape(length_pred.shape)    # reshape from (n,A) into (nA,)
            length_true_arr = torch.cat((length_true_arr,               # store all length labels in a linear array
                                         length_labels.to('cpu')),
                                        0)
    # calculate losses on entire batch
    stance_loss = sum(stance_losses) / len(stance_losses)
    length_loss = sum(length_losses) / len(length_losses)
    # remember to discard the 1st data point
    return stance_pred_arr[1:], stance_true_arr[1:], \
            length_pred_arr[1:], length_true_arr[1:], \
            stance_loss.item(), length_loss.item()


def save_model():
    '''
    For saving all the model parameters, optimizer states
    '''
    logging.info('Saving best model')
    torch.save([model.state_dict(),
                stance_optimizer.state_dict(),
                length_optimizer.state_dict()], 
               model_savefile)
    return 

# Feed into the model and see if it is working correctly
'''
batch_idx, minibatch = next(enumerate(train_dataloader))
batch_idx, minibatch = next(enumerate(test_dataloader))
'''

if __name__ == '__main__':
    start_time = time.time()
    train_losses = train(N_EPOCHS)
    torch.save(train_losses, model_savefile)    # save the losses
    
    train_stance_losses = train_losses[0]
    train_length_losses = train_losses[1]
    train_horz_index = train_losses[2]
    dev_stance_losses = train_losses[3]
    dev_length_losses = train_losses[4]
    dev_f1_scores = train_losses[5]
    dev_horz_index = train_losses[6]
    
    test_losses = test(mode='test')
    pred_stances = test_losses[0] 
    true_stances = test_losses[1] 
    pred_lengths = test_losses[2]
    true_lengths = test_losses[3]
    #loss_stance = test_losses[4]
    #loss_length = test_losses[5]
    
    stance_metrics = helper.stance_f1(pred_stances, 
                                      true_stances)
    length_metrics = helper.length_f1(pred_lengths, 
                                      true_lengths)
    
    time_end = time.time()
    time_taken = time_end - start_time  
    logger.info('Time elapsed: %6.2fs' % time_taken)
    fig,axes = plt.subplots(3,1)
    fig.show()
    ax1,ax2,ax3 = axes[0], axes[1], axes[2]
    
    ax1.set_title('stance loss')
    ax1.scatter(train_horz_index, train_stance_losses, color='red', s=10)
    ax1.scatter(dev_horz_index, dev_stance_losses, color='blue')
    ax1.set_yscale('log')
    
    ax2.set_title('length loss')
    ax2.scatter(train_horz_index, train_length_losses, color='red', s=10)
    ax2.scatter(dev_horz_index, dev_length_losses, color='blue')
    ax2.set_yscale('log')
    
    ax3.set_title('f1 score')
    ax3.scatter(dev_horz_index, dev_f1_scores)
    ax3.set_xlabel('minibatches')
    xlim = ax2.get_xlim()   # for setting the graphs x-axis to be the same
    ax3.set_xlim(xlim)      # for setting the graphs x-axis to be the same
    for each_axis in axes:
        each_axis.grid(True)
        each_axis.grid(True,which='minor')
    
    fig.set_size_inches(6, 8)
    fig.tight_layout()
    fig.savefig(plotfile)
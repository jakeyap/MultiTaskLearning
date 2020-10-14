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

'''======== FILE NAMES FOR LOGGING ========'''
FROM_SCRATCH = True
'''======== DEFAULT HYPERPARAMETERS ========'''
BATCH_SIZE_TRAIN = 2
LOG_INTERVAL = 1
N_EPOCHS = 1
LEARNING_RATE = 0.0005
MOMENTUM = 0.25
MAX_POST_LENGTH = 256
MAX_POST_PER_THREAD = 4
THREAD_LENGTH_DIVIDER = 9 # for dividing thread lengths into binary buckets
'''======== HYPERPARAMETERS END ========'''

def main():
    parser = argparse.ArgumentParser()
    global BATCH_SIZE_TRAIN, LOG_INTERVAL, N_EPOCHS, LEARNING_RATE, MOMENTUM
    global MAX_POST_LENGTH, MAX_POST_PER_THREAD, THREAD_LENGTH_DIVIDER
    # Grab the hyperparameters
    parser.add_argument("--BATCH_SIZE_TRAIN",   default=BATCH_SIZE_TRAIN,
                        type=int,               help="Minibatch size for training.")
    parser.add_argument("--LOG_INTERVAL",   default=LOG_INTERVAL,
                        type=int,           help="Num of minibatches before printing")
    parser.add_argument("--N_EPOCHS",       default=N_EPOCHS,
                        type=int,           help="Num of training epochs")
    parser.add_argument("--LEARNING_RATE",  default=LEARNING_RATE,
                        type=float,         help="learning rate for Adam.")
    parser.add_argument("--MOMENTUM",       default=MOMENTUM,
                        type=float,         help="momentum term for SGD.")
    parser.add_argument("--MAX_POST_LENGTH",    default=MAX_POST_LENGTH,    type=int,
                        help="Max input sequence length after BERT tokenizer")
    parser.add_argument("--MAX_POST_PER_THREAD",    default=MAX_POST_PER_THREAD,    type=int,
                        help="Max number of posts per thread to look at")
    parser.add_argument("--THREAD_LENGTH_DIVIDER",   default=THREAD_LENGTH_DIVIDER,
                        type=int,           help="Number to divide lengths into binary classes")
    
    parser.add_argument("--do_train",       action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",        action='store_true',
                        help="Whether to run eval on the dev set.")

    args = parser.parse_args()
    BATCH_SIZE_TRAIN = args.BATCH_SIZE_TRAIN
    LOG_INTERVAL = args.LOG_INTERVAL
    N_EPOCHS = args.N_EPOCHS
    LEARNING_RATE = args.LEARNING_RATE
    MOMENTUM = args.MOMENTUM
    MAX_POST_LENGTH = args.MAX_POST_LENGTH
    MAX_POST_PER_THREAD = args.MAX_POST_PER_THREAD
    THREAD_LENGTH_DIVIDER = args.THREAD_LENGTH_DIVIDER

    directory = './data/combined/'
    test_filename = 'encoded_shuffled_test_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    train_filename = 'encoded_shuffled_train_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    dev_filename = 'encoded_shuffled_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    
    '''
    test_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    train_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    dev_filename = 'encoded_combined_dev_%d_%d.pkl' % (MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    '''
    
    timestamp = time.time()
    timestamp = str(timestamp)[5:]
    
    suffix = "%d_%d_%d_%d_" % (BATCH_SIZE_TRAIN, N_EPOCHS, MAX_POST_PER_THREAD, MAX_POST_LENGTH)
    suffix = suffix + str(LEARNING_RATE) + "_"
    
    model_savefile = './saved_models/ModelA0_'+suffix+timestamp+'.bin'
    
    # for storing logs into a file
    file_handler = logging.FileHandler(filename='./log_files/ModelA0_'+suffix+timestamp+'.log')
    # for printing onto terminal
    stdout_handler = logging.StreamHandler(sys.stdout)
    
    # for storing training / dev / test losses
    lossfile = './log_files/losses_'+suffix+timestamp+'.bin'
    plotfile = './log_files/plot_'+suffix+timestamp+'.png'
    
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
    logger.info('LEARNING_RATE: %1.6f' % LEARNING_RATE)
    logger.info('MOMENTUM: %1.6f' % MOMENTUM)
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
            dev_loss_stance = test_results[4]               # extract the losses
            dev_loss_length = test_results[5]
            dev_f1_stance = test_results[6]                 # extract the f1 scores
            dev_f1_length = test_results[7]
            
            f1_new = (dev_f1_stance + dev_f1_length) / 2    # average the 2 f1 scores
            
            # store the dev set losses
            dev_stance_losses.append(dev_loss_stance)
            dev_length_losses.append(dev_loss_length)
            dev_horz_index.append(train_horz_index[-1])
            dev_f1_scores.append(f1_new)
            
            if (f1_new > best_f1_score):        # if f1 score beats previous scores
                best_f1_score = f1_new          # update the score
                save_model()                    # save the model
            
        return train_stance_losses, train_length_losses, train_horz_index, \
                dev_stance_losses, dev_length_losses, dev_f1_scores, dev_horz_index
    
    def test(mode='test', display=True):
        '''
        mode = 'test'|'dev'. Switch between test or validation modes
        display = True|False. Whether to display F1 metrics
        '''
        # Set network into evaluation mode
        model.eval()
        
        # Start the arrays to store the entire test set. 
        # Initialize a blank 1st data point first. Remember to delete later    
        stance_logits_arr = torch.zeros(size=(1,MAX_POST_PER_THREAD,5),dtype=torch.float)
        stance_labels_arr = torch.zeros(size=(1,MAX_POST_PER_THREAD),dtype=torch.int64)
        length_logits_arr = torch.zeros(size=(1,2),dtype=torch.float)
        length_labels_arr = torch.zeros(size=(1,1),dtype=torch.int64)
        
        if mode=='test':
            dataloader = test_dataloader
        elif mode=='dev':
            dataloader = dev_dataloader
        
        length_losses = []  # for tracking minibatch losses
        stance_losses = []  # for tracking minibatch losses
        
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(dataloader):
                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(('\tTesting '+ mode +' set Minibatch: %4d' % batch_idx))
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
                length_logits = model(input_ids = encoded_comments,         # get the length prediction logits
                                      token_type_ids = token_type_ids,      # (n,2): n=minibatch, 2=num of classes
                                      attention_masks = attention_masks, 
                                      task='length')
                
                stance_logits_arr = torch.cat((stance_logits_arr,           # store stance logits in a big linear array (N,A,B)
                                               stance_logits.to('cpu')),
                                              dim=0)
                length_logits_arr = torch.cat((length_logits_arr,           # store length logits in a big linear array (N,2)
                                               length_logits.to('cpu')),
                                              dim=0)
                
                stance_labels_arr = torch.cat((stance_labels_arr,           # store all stance labels in a big linear array (NA,1)
                                               stance_labels.to('cpu').long()),
                                              dim=0)
                length_labels_arr = torch.cat((length_labels_arr,           # store all length labels in a big linear array (N,1))
                                               length_labels.to('cpu').long()),
                                              dim=0)
                
            # Discarding the blank 1st data point.
            stance_logits_arr = stance_logits_arr[1:,:,:]   # shape was (n+1,A,5)
            stance_labels_arr = stance_labels_arr[1:,:,]    # shape was (n+1,A)
            length_logits_arr = length_logits_arr[1:,:]     # shape was (n+1,2)
            length_labels_arr = length_labels_arr[1:,:]     # shape was (n+1,1)
            
            stance_loss = helper.stance_loss(pred_logits=stance_logits_arr, # calculate the dev set stance loss
                                             true_labels=stance_labels_arr, 
                                             loss_fn=stance_loss_fn)
            
            length_loss = helper.length_loss(pred_logits=length_logits_arr, # calculate the dev set length loss
                                             true_labels=length_labels_arr, 
                                             loss_fn=length_loss_fn,
                                             divide=THREAD_LENGTH_DIVIDER)
            
            # convert everything into linear tensors
            stance_pred = helper.logit_2_class_stance(stance_logits_arr)    # convert logits to stance labels. (nA,)
            length_pred = helper.logit_2_class_length(length_logits_arr)    # convert logits to length labels. (n,)
            stance_true = stance_labels_arr.reshape(stance_pred.shape)      # reshape from (n,A) into (nA,)
            length_true = length_labels_arr.reshape(length_pred.shape)      # reshape from (n,1) into (n,)
            
            stance_metrics = helper.stance_f1(stance_pred,                  # calculate the f1-metrics for stance
                                              stance_true)
            length_metrics = helper.length_f1(length_pred,                  # calculate the f1-metrics for length
                                              length_true,
                                              THREAD_LENGTH_DIVIDER)
            
            stance_msg = helper.stance_f1_msg(stance_metrics[0],            # Get the strings to display for f1 scores
                                              stance_metrics[1],
                                              stance_metrics[2],
                                              stance_metrics[3],
                                              stance_metrics[4])
            length_msg = helper.length_f1_msg(length_metrics[0],            # Get the strings to display for f1 scores
                                              length_metrics[1],
                                              length_metrics[2],
                                              length_metrics[3],
                                              length_metrics[4])
            
            f1_stance_macro = stance_metrics[4]
            f1_length_macro = length_metrics[4]
            
            if display:
                logger.info('\n'+stance_msg)
                logger.info('\n'+length_msg)
        
        return stance_pred, stance_true, length_pred, length_true, \
                stance_loss.item(), length_loss.item(), \
                f1_stance_macro, f1_length_macro
    
    
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
    
    start_time = time.time()
    
    train_losses = train(N_EPOCHS)              
    torch.save(train_losses, model_savefile)    # save the losses
    test_losses = test(mode='test')
    
    train_stance_losses = train_losses[0]
    train_length_losses = train_losses[1]
    train_horz_index = train_losses[2]
    dev_stance_losses = train_losses[3]
    dev_length_losses = train_losses[4]
    dev_f1_scores = train_losses[5]
    dev_horz_index = train_losses[6]
    
    test_f1_stance = test_losses[6]
    test_f1_length = test_losses[7]
    test_f1_score = (test_f1_stance + test_f1_length) / 2 # average the 2
    
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
    ax3.scatter(dev_horz_index, dev_f1_scores, color='red')
    ax3.scatter(dev_horz_index[-1], test_f1_score, color='blue')
    ax3.set_xlabel('minibatches')
    xlim = ax2.get_xlim()   # for setting the graphs x-axis to be the same
    ax3.set_xlim(xlim)      # for setting the graphs x-axis to be the same
    for each_axis in axes:
        each_axis.grid(True)
        each_axis.grid(True,which='minor')
    
    fig.set_size_inches(6, 8)
    fig.tight_layout()
    fig.savefig(plotfile)

if __name__ == '__main__':
    main()
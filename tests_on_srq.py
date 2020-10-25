#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:24:23 2020
This file contains the logic to run tests on SRQ dataset
@author: jakeyap
"""


import DataProcessor
import multitask_helper_functions as helper
from classifier_models import my_ModelA0, my_ModelBn
import torch

import logging, sys, argparse
import time

def main():
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--MODELDIR',       default='./saved_models/',
                            help='Directory of model weights')
        parser.add_argument('--MODELFILE',      default='ModelA0_exp4.bin',
                            help='Filename of model weights')
        parser.add_argument('--DATADIR',        default='./data/srq/',
                            help='Directory for data')
        parser.add_argument('--DATAFILE',       default='encoded_stance_dataset_processed_4_256.pkl',
                            help='Datafile containing tokenized samples')
        
        parser.add_argument('--BATCH_SIZE_TEST',default=2,  type=int,
                            help='Minibatch size')
        parser.add_argument('--LOG_INTERVAL',   default=1,  type=int,
                            help='Num of minibatches before printing')
        
        parser.add_argument('--STANCE_NUM_LABELS',  default=5,  type=int,
                            help='Num of different stance classes')
        parser.add_argument('--LENGTH_NUM_LABELS',  default=2,  type=int,
                            help='Num of different length classes')
        parser.add_argument('--MAX_POST_LENGTH',    default=256, type=int,
                            help='Max input sequence length after BERT tokenizer')
        parser.add_argument('--MAX_POST_PER_THREAD', default=4,  type=int,
                            help='Max number of posts per thread to look at')
        
        parser.add_argument('--DEBUG',  action='store_true',
                            help='Set to true when debugging code')
        
        args = parser.parse_args()
        MODELDIR = args.MODELDIR                # directory of old model
        MODELFILE = args.MODELFILE              # filename of stored model
        DATADIR = args.DATADIR                  # directory of dataset
        DATAFILE = args.DATAFILE                # filename of test data
        
        BATCH_SIZE_TEST = args.BATCH_SIZE_TEST  # minibatch size (test)
        LOG_INTERVAL = args.LOG_INTERVAL        # how often to print
        
        STANCE_NUM_LABELS = args.STANCE_NUM_LABELS
        LENGTH_NUM_LABELS = args.LENGTH_NUM_LABELS
        MAX_POST_LENGTH = args.MAX_POST_LENGTH
        MAX_POST_PER_THREAD = args.MAX_POST_PER_THREAD
        
        DEBUG = args.DEBUG                      # debug flag
        
    logfile_name = './log_files/test_srq_'+MODELFILE[:-4]+'.log'    # save the log
    
    file_handler = logging.FileHandler(filename=logfile_name)       # for saving into a log file
    stdout_handler = logging.StreamHandler(sys.stdout)              # for printing onto terminal
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt= '%m/%d/%Y %H:%M:%S', handlers=handlers, level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info('Getting test data from %s' % DATADIR+DATAFILE)
    dataframe = DataProcessor.load_from_pkl(DATADIR+DATAFILE)       # get test data as dataframe
    if DEBUG:
        dataframe = dataframe[0:10]
    dataloader = DataProcessor.dataframe_2_dataloader(dataframe,    # pack data into dataloader
                                                      batchsize=BATCH_SIZE_TEST,
                                                      randomize=False,
                                                      DEBUG=DEBUG,
                                                      num_workers=5)
    
    logger.info('Getting model ready')
    model = get_model(MODELDIR, MODELFILE, 
                      stance_num_labels=STANCE_NUM_LABELS, 
                      length_num_labels=LENGTH_NUM_LABELS, 
                      max_post_num=MAX_POST_PER_THREAD,
                      max_post_len=MAX_POST_LENGTH)
    
    results =  test(model, dataloader, LOG_INTERVAL, -1)
    stance_pred = results[0]    # shape is (NA,)
    stance_true = results[1]    # shape is (NA,)
    
    # select only the replies to do analysis on
    # SRQ dataset only has parent, 1 reply. ignore all other labels
    start=1
    stop = stance_pred.shape[0]
    step = MAX_POST_PER_THREAD
    index_2_pick = range(start,stop,step)
    
    stance_pred = stance_pred[index_2_pick]
    stance_true = stance_true[index_2_pick]
    
    stance_accuracy = helper.accuracy_stance(stance_pred,   # calculate prediction accuracies
                                             stance_true)
    
    stance_metrics = helper.stance_f1(stance_pred,          # calculate the f1-metrics for stance
                                      stance_true,
                                      incl_empty=False)
    
    stance_msg = helper.stance_f1_msg(stance_metrics[0],    # Get the strings to display for f1 scores
                                      stance_metrics[1],
                                      stance_metrics[2],
                                      stance_metrics[3],
                                      stance_metrics[4],
                                      incl_empty=False)
    logger.info('\n'+stance_msg)
    logger.info('Stance Accuracy: %1.4f' % stance_accuracy)
    return stance_pred, stance_true


def test(model, dataloader, log_interval, index=-1):
    # if index==-1, just test all
    # if index==-2, just randomly test on 1 particular sentence, then print it
    # if index==any other number, find the post with that index and test it
    logger = logging.getLogger(__name__)
    
    model.eval() # set model to test mode first
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    # Start the arrays to store entire dataset
    stance_logits_arr = None
    stance_labels_arr = None
    
    with torch.no_grad():
        for batch_id, minibatch in enumerate(dataloader):
            if batch_id % log_interval == 0:
                logger.info(('\tTesting SRQ Minibatch: %4d' % batch_id))
            
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
            stance_logits = model(input_ids = encoded_comments,         # shape (n,A,B) where 
                                  token_type_ids = token_type_ids,      # n: minibatch
                                  attention_masks = attention_masks,    # A: num posts per thread
                                  task='stance')                        # B: num of stance classes 
            # no need to test length cauz SRQ lengths are all 2
            
            if stance_logits_arr is None:
                stance_logits_arr = stance_logits.to(cpu)               # for handling first minibatch only
                stance_labels_arr = stance_labels.to(cpu).long()
            else:
                stance_logits_arr = torch.cat((stance_logits_arr,       # shape (N,A,B)
                                               stance_logits.to(cpu)),  # N is entire threads length
                                              0)
                stance_labels_arr = torch.cat((stance_labels_arr,       # shape is (NA,1)
                                               stance_labels.to(cpu).long()),
                                              0)
            
        stance_pred =helper.logit_2_class_stance(stance_logits_arr) # convert logits to stance labels. (NA,)
        stance_true = stance_labels_arr.reshape(stance_pred.shape)  # reshape from (N,A) into (NA,)
        
    return [stance_pred, stance_true]


def get_model(modeldir, modelfile, 
              stance_num_labels=5, 
              length_num_labels=2,
              max_post_num=4,
              max_post_len=256):
    ''' Returns the model, with weights loaded '''
    logger = logging.getLogger(__name__)
    if 'modela0'==modelfile.lower()[0:7]:
        model = my_ModelA0.from_pretrained('bert-base-uncased',
                                           stance_num_labels=stance_num_labels,
                                           length_num_labels=length_num_labels,
                                           max_post_num=max_post_num, 
                                           max_post_length=max_post_len)
    elif 'modelb'==modelfile.lower()[0:6]:
        number = int(modelfile[6])
        model = my_ModelBn.from_pretrained('bert-base-uncased',
                                           stance_num_labels=stance_num_labels,
                                           length_num_labels=length_num_labels,
                                           max_post_num=max_post_num, 
                                           max_post_length=max_post_len,
                                           num_transformers=number)
    else:
        logger.info('Exiting, model not found: ' + modelfile)
        raise Exception
    
    model.resize_token_embeddings(len(DataProcessor.default_tokenizer))
    model = model.cuda()                    # put the model into GPU
    model = torch.nn.DataParallel(model)    # wrap the model using DataParallel
    
    temp = torch.load(modeldir+modelfile)   # load the best model
    model_state = temp[0]                   # get the model state
    model.load_state_dict(model_state)      # stuff state into model
    return model

if __name__ == '__main__':
    main()
    
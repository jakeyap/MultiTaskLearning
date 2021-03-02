#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:49:56 2021
This file is for training on tree structured data.
The data is inside ./data/coarse_discourse/full_trees/

@author: jakeyap
"""

import torch
import torch.optim as optim
import numpy as np

import TreeDataProcessor
import multitask_helper_functions as helper
from classifier_models_v2 import alt_ModelFn, alt_ModelGn, SelfAdjDiceLoss
from utilities.handle_coarse_discourse_trees import RedditTree

import logging, sys, argparse
import time
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

def main():
    '''======== DEFAULT HYPERPARAMETERS ========'''
    if True: # Collapse this for readability
        # Grab the non default hyperparameters from input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--BATCH_SIZE_TRAIN",   default=2,
                            type=int,               help="Minibatch size for training.")
        parser.add_argument("--BATCH_SIZE_TEST",   default=2,
                            type=int,               help="Minibatch size for training.")
        parser.add_argument("--LOG_INTERVAL",   default=1,
                            type=int,           help="Num of minibatches before printing")
        parser.add_argument("--N_EPOCHS",       default=1,
                            type=int,           help="Num of training epochs")
        parser.add_argument("--LEARNING_RATE",  default=0.0005,
                            type=float,         help="learning rate for Adam.")
        parser.add_argument("--MOMENTUM",       default=0.25,
                            type=float,         help="momentum term for SGD.")
        parser.add_argument("--MAX_POST_LENGTH",    default=256,    type=int,
                            help="Max input sequence length after BERT tokenizer")
        parser.add_argument("--STRIDES",        default=1,    type=int,
                            help="Number of horz strides of kids to check")
        parser.add_argument("--THREAD_LENGTH_DIVIDER",   default=9,
                            type=int,           help="Number to divide lengths into binary classes")
        parser.add_argument("--OPTIMIZER",      default='SGD',
                            help="SGD (Default) or ADAM.")
        
        parser.add_argument("--WEIGHTED_STANCE",action='store_true',
                            help="Whether to weigh DENY and QUERY higher")
        parser.add_argument("--DOUBLESTEP",     action='store_true',
                            help="Whether to double step length training")
        
        parser.add_argument("--DO_TRAIN",       action='store_true',
                            help="Whether to run training. Not implemented yet")
        parser.add_argument("--DO_TEST",        action='store_true',
                            help="Whether to run eval on the test set. Not implemented yet")
        parser.add_argument("--DEBUG",          action='store_true',
                            help="Debug flag")
        parser.add_argument("--MODELNAME",          default='ModelA0',
                            help="For choosing model")
        parser.add_argument("--NAME",          default='',
                            help="For naming log files")
    '''======== HYPERPARAMETERS END ========'''
    
    args = parser.parse_args()
    BATCH_SIZE_TRAIN = args.BATCH_SIZE_TRAIN            # minibatch size (train)
    BATCH_SIZE_TEST = args.BATCH_SIZE_TEST              # minibatch size (test)
    LOG_INTERVAL = args.LOG_INTERVAL                    # how often to print
    N_EPOCHS = args.N_EPOCHS                            # number of epochs to train
    LEARNING_RATE = args.LEARNING_RATE                  # learning rate
    MOMENTUM = args.MOMENTUM                            # momentum for SGD
    MAX_POST_LENGTH = args.MAX_POST_LENGTH              # num of tokens per post 
    
    STRIDES = args.STRIDES                              # number of groups of kids to inspect per root
    THREAD_LENGTH_DIVIDER = args.THREAD_LENGTH_DIVIDER  # how to split the dataset for binary cls
    OPTIM_NAME = args.OPTIMIZER                         # ADAM, SGD or RMSProp
    WEIGHTED_STANCE = args.WEIGHTED_STANCE              # Whether to weigh cost functions or flat
    DOUBLESTEP = args.DOUBLESTEP                        # Whether to double step length training
    
    DO_TRAIN = args.DO_TRAIN                            # Whether to run training
    DO_TEST = args.DO_TEST                              # Whether to run a test
    
    DEBUG = args.DEBUG
    NAME = args.NAME
    MODELNAME = args.MODELNAME
    
    timestamp = time.time()
    if NAME=='':
        expname = str(timestamp)[5:]
    else:
        expname = NAME
    
    model_savefile = './saved_models/'+MODELNAME+'_'+expname+'.bin'
    
    if DO_TRAIN:    # to store training logs into a file
        logfile_name = './log_files/'+MODELNAME+'_'+expname+'.log'
    else:           # if just doing testing, save to a different place
        logfile_name = './log_files/'+MODELNAME+'_'+expname+'.testlog'
    
    file_handler = logging.FileHandler(filename=logfile_name)           # for printing into logfile
    stdout_handler = logging.StreamHandler(sys.stdout)                  # for printing onto terminal
    
    lossfile = './log_files/losses_'+MODELNAME+'_'+expname+'.bin'       # to store train/dev/test losses
    plotfile = './log_files/plot_'+MODELNAME+'_'+expname+'.png'         # to plot losses
    
    handlers = [file_handler, stdout_handler]                           # stuff to handle loggers
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        handlers=handlers,
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    
    gpu = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    torch.cuda.empty_cache()
    
    if MODELNAME.lower()[:-1]=='alt_modelf':
        number = int(MODELNAME[-1])
        model = alt_ModelFn.from_pretrained('bert-base-uncased', 
                                            max_post_length=MAX_POST_LENGTH,
                                            num_transformers=number)
    elif MODELNAME.lower()[:-1]=='alt_modelg':
        number = int(MODELNAME[-1])
        model = alt_ModelGn.from_pretrained('bert-base-uncased',
                                            max_post_length=MAX_POST_LENGTH,
                                            num_transformers=number)
    else:
        logger.info('Exiting, model not found: ' + MODELNAME)
        raise Exception()
    '''
    # bert base large is too large to fit into GPU RAM. discuss with team
    '''
    # Resize model vocab
    model.resize_token_embeddings(len(TreeDataProcessor.default_tokenizer))
    
    # Move model into GPU
    model.to(gpu)
    
    # Define the optimizers. 
    if OPTIM_NAME=='SGD':            # Use SGD
        logger.info('Using SGD')
        stance_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                     momentum=MOMENTUM)
        length_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                     momentum=MOMENTUM)
    elif OPTIM_NAME=='ADAM':         # Use ADAM
        logger.info('Using Adam')
        stance_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if 'modelc' in MODELNAME.lower():
            length_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 3)
        else:
            length_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIM_NAME=='ADAM_V':
        stance_optimizer = optim.Adam([{'params' : model.bert.parameters()},
                                       {'params' : model.transformer_stance.parameters(), 'lr' : LEARNING_RATE},
                                       {'params' : model.stance_classifier.parameters(), 'lr' : LEARNING_RATE}], 
                                      lr=1e-5)
        length_optimizer = optim.Adam([{'params' : model.bert.parameters()},
                                       {'params' : model.transformer_length.parameters(), 'lr' : LEARNING_RATE},
                                       {'params' : model.length_classifier.parameters(), 'lr' : LEARNING_RATE}], 
                                      lr=1e-5)
    else:
        logger.info('Exiting. No such optimizer '+OPTIM_NAME)
        raise Exception()
    
    logger.info('Running on %d GPUs' % n_gpu)
    #if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    
    # Set up loss functions. Use averaging to calculate a value
    if WEIGHTED_STANCE:
        # increase the weights for disagreement and -ve reaction
        # weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        #''' FOR EXP 5-8 ONLY. REVERSE PUNISH EMPTY, QUESTION, ANSWER'''
        weights = torch.tensor([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(gpu)
        #''' FOR EXP 47-50 ONLY '''
        #weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]).to(gpu)
        #stance_loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        stance_loss_fn = SelfAdjDiceLoss(reduction='mean')
    else:
        #stance_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        stance_loss_fn = SelfAdjDiceLoss(reduction='mean')
    #length_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    length_loss_fn = SelfAdjDiceLoss(reduction='mean')
    logger.info('======== '+MODELNAME+' =========')
    logger.info('===== Hyperparameters ======')
    logger.info('BATCH_SIZE_TRAIN: %d' % BATCH_SIZE_TRAIN)
    logger.info('LEARNING_RATE: %1.6f' % LEARNING_RATE)
    logger.info('MOMENTUM: %1.6f' % MOMENTUM)
    logger.info('MAX_POST_LENGTH: %d' % MAX_POST_LENGTH)
    logger.info('THREAD_LENGTH_DIVIDER: %d' % THREAD_LENGTH_DIVIDER)
    
    logger.info('===== Other settings ======')
    logger.info('OPTIMIZER: ' + OPTIM_NAME)
    logger.info('WEIGHTED STANCE LOSS' if WEIGHTED_STANCE else 'FLAT STANCE LOSS')
    if DOUBLESTEP:
        logger.info('DOUBLESTEPPING LENGTH')
    
    # Load data into dataframes
    logger.info('====== Loading dataframes ========')
    df_test, df_eval, df_trng = TreeDataProcessor.get_df_strat_1(max_post_len=MAX_POST_LENGTH,
                                                                 strides=STRIDES,
                                                                 DEBUG=DEBUG)
    
    # Pack dataframes into dataloaders
    logger.info('Converting dataframes into dataloaders')
    trng_dl = TreeDataProcessor.df_2_dataloader(df_trng,
                                                batchsize=BATCH_SIZE_TRAIN,
                                                randomize=True,
                                                DEBUG=DEBUG,
                                                num_workers=4*n_gpu)
    test_dl = TreeDataProcessor.df_2_dataloader(df_test,
                                                batchsize=BATCH_SIZE_TEST,
                                                randomize=False,
                                                DEBUG=DEBUG,
                                                num_workers=4*n_gpu)
    eval_dl = TreeDataProcessor.df_2_dataloader(df_eval,
                                                batchsize=BATCH_SIZE_TEST,
                                                randomize=False,
                                                DEBUG=DEBUG,
                                                num_workers=4*n_gpu)
    
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
            for batch_idx, minibatch in enumerate(trng_dl):
                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch_counter, batch_idx)))
                
                # get the features from dataloader
                # post_index = minibatch[0] # shape = (n, 1). unused
                input_ids = minibatch[1]    # shape = (n, 7 x num_tokens_per_post)
                token_types = minibatch[2]  # shape = (n, 7 x num_tokens_per_post)
                att_masks = minibatch[3]    # shape = (n, 7 x num_tokens_per_post)
                labels_arr = minibatch[4]   # shape = (n, 7)
                tree_size = minibatch[5]    # shape = (n, 1)
                # fam_size = minibatch[6]   # shape = (n, 1). unused
                
                # move features to gpu
                input_ids = input_ids.to(gpu)
                token_types = token_types.to(gpu)
                att_masks = att_masks.to(gpu)
                length_labels = tree_size.to(gpu)
                stance_labels = labels_arr.to(gpu)
                
                if DOUBLESTEP:    # try double stepping
                    # get the length prediction logits
                    length_logits = model(input_ids = input_ids,
                                          token_type_ids = token_types, 
                                          attention_masks = att_masks, 
                                          task='length')
                    # calculate the length loss
                    length_loss = helper.length_loss(pred_logits = length_logits,
                                                     true_labels = length_labels, 
                                                     loss_fn = length_loss_fn,
                                                     divide = THREAD_LENGTH_DIVIDER)
                    length_loss.backward()              # back propagation
                    length_optimizer.step()             # step the gradients once
                    length_optimizer.zero_grad()        # clear the gradients before the next step
                
                # get the stance prediction logits
                stance_logits = model(input_ids = input_ids,
                                      token_type_ids = token_types, 
                                      attention_masks = att_masks, 
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
                length_logits = model(input_ids = input_ids,
                                      token_type_ids = token_types, 
                                      attention_masks = att_masks, 
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
                train_horz_index.append(epoch_counter * len(trng_dl) + batch_idx)
                
            # TODO REACHED HERE
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
        stance_logits_arr = torch.zeros(size=(1,7,11),dtype=torch.float)
        stance_labels_arr = torch.zeros(size=(1,7),dtype=torch.int64)
        length_logits_arr = torch.zeros(size=(1,2),dtype=torch.float)
        length_labels_arr = torch.zeros(size=(1,1),dtype=torch.int64)
        
        if mode=='test':
            dataloader = test_dl
        elif mode=='dev':
            dataloader = eval_dl
        
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(dataloader):
                if batch_idx % LOG_INTERVAL == 0:
                    logger.info(('\tTesting '+ mode +' set Minibatch: %4d' % batch_idx))
                
                # get the features from dataloader
                # post_index = minibatch[0] # shape = (n, 1). unused
                input_ids = minibatch[1]    # shape = (n, 7 x num_tokens_per_post)
                token_types = minibatch[2]  # shape = (n, 7 x num_tokens_per_post)
                att_masks = minibatch[3]    # shape = (n, 7 x num_tokens_per_post)
                labels_arr = minibatch[4]   # shape = (n, 7)
                tree_size = minibatch[5]    # shape = (n, 1)
                # fam_size = minibatch[6]   # shape = (n, 1). unused
                
                # move features to gpu
                input_ids = input_ids.to(gpu)
                token_types = token_types.to(gpu)
                att_masks = att_masks.to(gpu)
                length_labels = tree_size.to(gpu)
                stance_labels = labels_arr.to(gpu)
                
                # get the stance prediction logits
                stance_logits = model(input_ids = input_ids,
                                      token_type_ids = token_types, 
                                      attention_masks = att_masks, 
                                      task='stance')
                
                # calculate the stance loss
                stance_loss = helper.stance_loss(pred_logits = stance_logits,
                                                 true_labels = stance_labels, 
                                                 loss_fn = stance_loss_fn)
                
                # get the length prediction logits
                length_logits = model(input_ids = input_ids,
                                      token_type_ids = token_types, 
                                      attention_masks = att_masks, 
                                      task='length')
                # calculate the length loss
                length_loss = helper.length_loss(pred_logits = length_logits,
                                                 true_labels = length_labels, 
                                                 loss_fn = length_loss_fn,
                                                 divide = THREAD_LENGTH_DIVIDER)
                
                # =============================                
                stance_logits_arr = torch.cat((stance_logits_arr,           # store stance logits in a big linear array (N,7,11)
                                               stance_logits.to('cpu')),
                                              dim=0)
                length_logits_arr = torch.cat((length_logits_arr,           # store length logits in a big linear array (N,2)
                                               length_logits.to('cpu')),
                                              dim=0)
                
                stance_labels_arr = torch.cat((stance_labels_arr,           # store all stance labels in a big linear array (7N,1)
                                               stance_labels.to('cpu').long()),
                                              dim=0)
                length_labels_arr = torch.cat((length_labels_arr,           # store all length labels in a big linear array (N,1))
                                               length_labels.to('cpu').long()),
                                              dim=0)
                
            # Discarding the blank 1st data point.
            stance_logits_arr = stance_logits_arr[1:,:,:]   # shape was (n+1,7,11), now is (n,7,11)
            stance_labels_arr = stance_labels_arr[1:,:,]    # shape was (n+1,7), now is (n,7)
            length_logits_arr = length_logits_arr[1:,:]     # shape was (n+1,2), now is (n,2)
            length_labels_arr = length_labels_arr[1:,:]     # shape was (n+1,1), now is (n,1)
            
            stance_loss = helper.stance_loss(stance_logits_arr.to(gpu),     # calculate the dev set stance loss
                                             stance_labels_arr.to(gpu),     # move to GPU, cauz loss weights are in GPU
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
                                              stance_true,
                                              coarse_disc=True)
            '''
            st_metrics_noempty = helper.stance_f1(stance_pred,              # calculate the f1-metrics for stance wo isempty
                                                  stance_true,
                                                  incl_empty=False,
                                                  coarse_disc=True)
            '''
            length_metrics = helper.length_f1(length_pred,                  # calculate the f1-metrics for length
                                              length_true,
                                              THREAD_LENGTH_DIVIDER)
            
            stance_accuracy = helper.accuracy_stance(stance_pred,           # calculate prediction accuracies
                                                     stance_true,
                                                     incl_empty=True)
            
            length_accuracy = helper.accuracy_length(length_pred,           # calculate prediction accuracies
                                                     length_true,
                                                     THREAD_LENGTH_DIVIDER)
            
            stance_msg = helper.stance_f1_msg(stance_metrics[0],            # Get the strings to display for f1 scores
                                              stance_metrics[1],
                                              stance_metrics[2],
                                              stance_metrics[3],
                                              stance_metrics[4],
                                              coarse_disc=True)
            '''
            stance_msg2 = helper.stance_f1_msg(st_metrics_noempty[0],            # Get the strings to display for f1 scores
                                               st_metrics_noempty[1],
                                               st_metrics_noempty[2],
                                               st_metrics_noempty[3],
                                               st_metrics_noempty[4],
                                               incl_empty=False,
                                               coarse_disc=True)
            '''
            length_msg = helper.length_f1_msg(length_metrics[0],            # Get the strings to display for f1 scores
                                              length_metrics[1],
                                              length_metrics[2],
                                              length_metrics[3],
                                              length_metrics[4])
            
            f1_stance_macro = stance_metrics[4]
            f1_length_macro = length_metrics[4]
            
            if display:
                logger.info('\n'+stance_msg)
                '''logger.info('without ismpety')
                logger.info('\n'+stance_msg2)'''
                logger.info('Stance Accuracy: %1.4f' % stance_accuracy)
                logger.info('\n'+length_msg)
                logger.info('Length Accuracy: %1.4f' % length_accuracy)
        
        return stance_pred, stance_true, length_pred, length_true, \
                stance_loss.item(), length_loss.item(), \
                f1_stance_macro, f1_length_macro
    
    def save_model():
        '''
        For saving all the model parameters, optimizer states
        '''
        logging.info('Saving best model')
        # dont save optimizer states to reduce HDD usage
        torch.save(model.state_dict(),model_savefile)
        return 
    
    ''' 
    # for debugging purposes
    batch_idx, minibatch = next(enumerate(train_dataloader))
    batch_idx, minibatch = next(enumerate(test_dataloader))
    '''
    time1 = time.time()
    if DO_TRAIN:
        train_losses = train(N_EPOCHS)
        torch.save(train_losses, lossfile)  # save the losses
    else:
        train_losses = torch.load(lossfile) # load losses from archive
    
    temp = torch.load(model_savefile)       # load the best model
    #model_state = temp[0]                   # get the model state
    model_state = temp                      # unlike older implementations, optimizers not stored
    model.load_state_dict(model_state)      # stuff state into model
    test_losses = test(mode='test')         # run a test 
    
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
    
    
    fig,axes = plt.subplots(3,1)
    fig.show()
    ax1,ax2,ax3 = axes[0], axes[1], axes[2]
    
    ax1.set_title('stance loss')
    ax1.set_yscale('log')
    ax1.scatter(dev_horz_index, dev_stance_losses, color='blue')
    ax1.scatter(train_horz_index, train_stance_losses, color='red', s=10)
    
    ax2.set_title('length loss')
    ax2.set_yscale('log')
    ax2.scatter(dev_horz_index, dev_length_losses, color='blue')
    ax2.scatter(train_horz_index, train_length_losses, color='red', s=10)
    
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
    time2 = time.time()
    hours = (time2-time1) // 3600
    remain = (time2-time1) % 3600
    minutes = remain // 60
    seconds = remain % 60
    
    logger.info('Time taken: %dh %dm %2ds' % (hours, minutes, seconds))

if __name__ == '__main__':
    main()
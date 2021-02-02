#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:51:55 2021
This file contains the logic to run tests on SRQ dataset
but it is only for the wider models.

It is built on tests_on_srq_mixed.py, but stacks an extra MLP at the end to 
learn the 11 to 5 mapping instead of hardcoding. Train using just 1 epoch
@author: jakeyap
"""

import DataProcessor
import multitask_helper_functions as helper
from classifier_models_v2 import alt_ModelFn, final_mapping_layer

import torch
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

import logging, sys, argparse
import time

torch.manual_seed(0)
np.random.seed(0)

def main():
    start_time = time.time()
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--MODELDIR',       default='./saved_models/',
                            help='Directory of model weights')
        parser.add_argument('--MODELFILE',      default='ModelA0_exp4.bin',
                            help='Filename of model weights')
        parser.add_argument('--DATADIR',        default='./data/srq/',
                            help='Directory for data')
        parser.add_argument('--DATAFILE',       default='encoded_stance_dataset_processed_4_128.pkl',
                            help='Datafile containing tokenized samples')
        parser.add_argument('--BATCH_SIZE_TRAIN',default=2,  type=int,
                            help='Minibatch size')
        parser.add_argument('--BATCH_SIZE_TEST',default=2,  type=int,
                            help='Minibatch size')
        
        parser.add_argument("--N_EPOCHS",       default=1,
                            type=int,           help="Num of training epochs")
        parser.add_argument("--LEARNING_RATE",  default=0.001,
                            type=float,         help="learning rate for Adam.")
        parser.add_argument('--LOG_INTERVAL',   default=1,  type=int,
                            help='Num of minibatches before printing')
        
        parser.add_argument('--MAX_POST_LENGTH',    default=256, type=int,
                            help='Max input sequence length after BERT tokenizer')
        parser.add_argument('--MAX_POST_PER_THREAD', default=4,  type=int,
                            help='Max number of posts per thread to look at')
        
        parser.add_argument('--DEBUG',  action='store_true',
                            help='Set to true when debugging code')
        parser.add_argument('--MAP_OPTION', default=1, type=int,
                            help="Hard coded mapping options from coarse disc to SDQC labels")
        
        parser.add_argument('--TRAIN_PCT', default=10, type=int,
                            help="Percentage of dataset to train on")
        parser.add_argument('--DO_TRAIN', action='store_true',
                            help="To learn a mapping from coarse disc to SDQC labels")
        
        args = parser.parse_args()
        MODELDIR = args.MODELDIR                # directory of old model
        MODELFILE = args.MODELFILE              # filename of stored model
        DATADIR = args.DATADIR                  # directory of dataset
        DATAFILE = args.DATAFILE                # filename of test data
        
        BATCH_SIZE_TRAIN =args.BATCH_SIZE_TRAIN # minibatch size (test)
        BATCH_SIZE_TEST = args.BATCH_SIZE_TEST  # minibatch size (test)
        N_EPOCHS = args.N_EPOCHS                # number of training epochs
        LEARNING_RATE = args.LEARNING_RATE      # learning rate
        LOG_INTERVAL = args.LOG_INTERVAL        # how often to print
        
        MAX_POST_LENGTH = args.MAX_POST_LENGTH
        MAX_POST_PER_THREAD = args.MAX_POST_PER_THREAD
        DEBUG = args.DEBUG                      # debug flag
        MAP_OPTION = args.MAP_OPTION            # how to map from Coarse disc labels to SDQC labels
        TRAIN_PCT = args.TRAIN_PCT              # Percentage of dataset to train on
        DO_TRAIN = args.DO_TRAIN                # boolean to decide to train or not
        
        ADDONFILE = MODELFILE.replace('.bin','_addon.bin')  # filename of final mapping layer
        
    logfile_name = './log_files/test_srq_'+MODELFILE[:-4]+'_addon.log'  # save the log
    plotfile0 ='./log_files/plot_srq_'+MODELFILE[:-4]+'_addon.png'      # save the log
    plotfile1 ='./log_files/plot_srq_'+MODELFILE[:-4]+'_mapped.png'     # save the log
    results_name = './log_files/'+MODELFILE[:-4]+'_addon_predict.txt'   # save the log
    results_file = open(results_name,'w')
    
    file_handler = logging.FileHandler(filename=logfile_name)       # for saving into a log file
    stdout_handler = logging.StreamHandler(sys.stdout)              # for printing onto terminal
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt= '%m/%d/%Y %H:%M:%S', handlers=handlers, level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    logger.info('======== '+MODELFILE+' =========')
    logger.info('===== Hyperparameters ======')
    logger.info('BATCH_SIZE_TRAIN: %d' % BATCH_SIZE_TRAIN)
    logger.info('LEARNING_RATE: %1.6f' % LEARNING_RATE)
    logger.info('N_EPOCHS: %d ' % N_EPOCHS)
    
    logger.info('====== Loading dataframes ========')
    logger.info('Getting test data from %s' % DATADIR+DATAFILE)
    dataframe = DataProcessor.load_from_pkl(DATADIR+DATAFILE)       # get test data as dataframe
    
    if DEBUG:
        dataframe = dataframe[0:600]
    
    datalen = len(dataframe)
    logger.info('Data size: %d ' % datalen)
    trng_idx = []
    test_idx = []
    eval_idx = []
    for i in range (datalen):
        if i % 100 < TRAIN_PCT:
            trng_idx.append(i)
        elif (i % 100 >= TRAIN_PCT) and (i % 100 < TRAIN_PCT + 10) :
            eval_idx.append(i)
        else:
            test_idx.append(i)
    
    test_df = dataframe.iloc[test_idx]
    eval_df = dataframe.iloc[eval_idx]
    trng_df = dataframe.iloc[trng_idx]
    #test_df = dataframe.iloc[trng_idx]
    #eval_df = dataframe.iloc[trng_idx]
    #trng_df = dataframe.iloc[trng_idx]
    
    logger.info('Converting dataframes into dataloaders')
    test_dl = DataProcessor.dataframe_2_dataloader(test_df,     # pack data into dataloader
                                                   batchsize=BATCH_SIZE_TEST,
                                                   randomize=False,
                                                   num_workers=5)
    eval_dl = DataProcessor.dataframe_2_dataloader(eval_df,     # pack data into dataloader
                                                   batchsize=BATCH_SIZE_TEST,
                                                   randomize=False,
                                                   num_workers=5)
    trng_dl = DataProcessor.dataframe_2_dataloader(trng_df,     # pack data into dataloader
                                                   batchsize=BATCH_SIZE_TRAIN,
                                                   randomize=True,
                                                   num_workers=5)
    
    
    logger.info('Test file is '+MODELFILE)
    logger.info('Getting model ready')
    model = get_model(MODELDIR, MODELFILE, 
                      max_post_num=MAX_POST_PER_THREAD,
                      max_post_len=MAX_POST_LENGTH)
    model.cuda()
    
    # TODO : finish this function that trains the mapping at final layer
    if DO_TRAIN:
        addon = final_mapping_layer()
        addon.cuda()
        #weights = torch.tensor([0.1, 0.1, 0.1, 1.0, 0.1]).to('cuda')
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to('cuda')
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
        optimizer = optim.Adam(addon.parameters(), lr=LEARNING_RATE)
        filename = MODELDIR + ADDONFILE
        train(model=model, mappinglayer=addon, 
              trng_dl=trng_dl, 
              eval_dl=eval_dl,
              epochs=N_EPOCHS,
              log_interval=LOG_INTERVAL, 
              loss_fn=loss_fn,
              optimizer=optimizer,
              num_post=MAX_POST_PER_THREAD, 
              post_length=MAX_POST_LENGTH,
              filename=filename,
              plotfile=plotfile0)
    
    addon = get_addon(MODELDIR, ADDONFILE)
    addon.cuda()
    results =  test(model=model, 
                    mappinglayer=addon,
                    dataloader=test_dl, 
                    log_interval=LOG_INTERVAL, 
                    num_post=MAX_POST_PER_THREAD, 
                    post_length=MAX_POST_LENGTH)
    stance_pred = results[0]    # shape is (NA,)
    stance_true = results[1]    # shape is (NA,)
    premap_pred = results[2]    # shape is (NA,)
    
    stance_pred = stance_pred.to('cpu')
    stance_true = stance_true.to('cpu')
    
    # select only the replies to do analysis on
    # SRQ dataset only has parent, 1 reply. ignore all other labels
    start=1
    stop = stance_pred.shape[0]
    step = MAX_POST_PER_THREAD
    index_2_pick = range(start,stop,step)
    
    stance_pred = stance_pred[index_2_pick]
    stance_true = stance_true[index_2_pick]
    premap_pred = premap_pred[index_2_pick]
    
    print('Premapped')
    print('Empty\t%5d' % (premap_pred==0).sum())
    print('Quest\t%5d' % (premap_pred==1).sum())
    print('Answe\t%5d' % (premap_pred==2).sum())
    print('Annou\t%5d' % (premap_pred==3).sum())
    print('Agree\t%5d' % (premap_pred==4).sum())
    print('Appre\t%5d' % (premap_pred==5).sum())
    print('Disag\t%5d' % (premap_pred==6).sum())
    print('Neg.r\t%5d' % (premap_pred==7).sum())
    print('Elabo\t%5d' % (premap_pred==8).sum())
    print('Humor\t%5d' % (premap_pred==9).sum())
    print('Other\t%5d' % (premap_pred==10).sum())
    
    print('\nPredict')
    print('Empty\t%5d' % (stance_pred==0).sum())
    print('Deny \t%5d' % (stance_pred==1).sum())
    print('Suppo\t%5d' % (stance_pred==2).sum())
    print('Query\t%5d' % (stance_pred==3).sum())
    print('Comme\t%5d' % (stance_pred==4).sum())
    
    #weight = addon.linear1.weight
    #print(weight)
    
    # if modelD or modelE is used, need to convert the labels to 5 classes
    '''
    print(stance_pred.shape)
    logger.info('Predictions before mapping')
    logger.info('empty %4d' % (stance_pred == 0).sum())
    logger.info('quest %4d' % (stance_pred == 1).sum())
    logger.info('answe %4d' % (stance_pred == 2).sum())
    logger.info('annou %4d' % (stance_pred == 3).sum())
    logger.info('agree %4d' % (stance_pred == 4).sum())
    logger.info('appre %4d' % (stance_pred == 5).sum())
    logger.info('disag %4d' % (stance_pred == 6).sum())
    logger.info('neg.r %4d' % (stance_pred == 7).sum())
    logger.info('elabo %4d' % (stance_pred == 8).sum())
    logger.info('humor %4d' % (stance_pred == 9).sum())
    logger.info('other %4d' % (stance_pred == 10).sum())
    
    logger.info('Mapping predictions using option %d' % MAP_OPTION)
    stance_pred = map_coarse_discourse_2_sdqc_labels(stance_pred, 
                                                     option=MAP_OPTION)
    '''
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
    
    stance_true = helper.rescale_labels(stance_true)        # take care of -1 in labels first
    helper.plot_confusion_matrix(y_true = stance_true,
                                 y_pred = stance_pred, 
                                 labels=[1,2,3,4],
                                 label_names=['deny','support','query','comment'])
    plt.savefig(plotfile1)
    time_end = time.time()
    time_taken = time_end - start_time  
    logger.info('Time elapsed: %6.2fs' % time_taken)
    
    for i in range(len(stance_pred)):
        results_file.write(str(int(stance_true[i].item())) + ','+ 
                           str(int(stance_pred[i].item())) + '\n')
    return stance_pred, stance_true

def train(model, mappinglayer, trng_dl, eval_dl, epochs, log_interval, loss_fn, optimizer, 
          num_post=4, post_length=128, filename='', plotfile=''):
    #TODO: implement training on 10% of SRQ
    loss_values = []
    loss_horz = []
    f1_scores = []
    f1_horz = []
    best_f1 = -1
    
    logger = logging.getLogger(__name__)
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    for epoch in range(epochs):
        model.eval()            # set main model into test mode
        mappinglayer.train()    # set mapping layer into training mode
        weights_list = []
        
        for batch_id, minibatch in enumerate(trng_dl):
            weights_list.append(mappinglayer.linear1.weight.clone())
            if batch_id % log_interval == 0:
                logger.info(('\tEPOCH: %3d\tMiniBatch: %4d' % (epoch, batch_id)))
            encoded_comments = minibatch[1]
            token_type_ids = minibatch[2]
            attention_masks = minibatch[3]
            #length_labels = minibatch[4]
            stance_labels = minibatch[5]
            
            # if necessary, extend data to 7 post long if model expects more data
            if num_post > 4:
                mb = encoded_comments.shape[0]
                encoded_comments_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                encoded_comments_2[:,:post_length*4] = encoded_comments[:,:post_length*4] # extend to (n, however needed)
                encoded_comments = encoded_comments_2
                token_type_ids_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                token_type_ids_2[:,:post_length*4] = token_type_ids[:,:post_length*4] # extend to (n, however needed)
                token_type_ids = token_type_ids_2
                attention_masks_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                attention_masks_2[:,:post_length*4] = attention_masks[:,:post_length*4] # extend to (n, however needed)
                attention_masks = attention_masks_2
                
                stance_labels_2 = torch.ones((mb, num_post), dtype=torch.long) * -1
                stance_labels_2[:, :4] = stance_labels[:, :4]
                stance_labels = stance_labels_2
            
            # move features to gpu
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            with torch.no_grad():
                # get the stance prediction logits
                stance_logits = model(input_ids = encoded_comments,         # shape (n,7,B) where 
                                      token_type_ids = token_type_ids,      # n: minibatch
                                      attention_masks = attention_masks,    # 7: num posts per thread
                                      task='stance')                        # B: num of stance classes 
                
                # only calculate losses based on root post and 1 reply
                stance_labels = stance_labels[:, :2]    # orig shape=(n,7). new=(n,2)
                stance_logits = stance_logits[:, :2, :] # orig shape=(n,7,11). new=(n,2,11)
            mapped_logits = mappinglayer(stance_logits)
            
            # calculate the stance loss
            stance_loss = helper.stance_loss(pred_logits = mapped_logits,
                                             true_labels = stance_labels.to(gpu), 
                                             loss_fn = loss_fn)
            
            stance_loss.backward()      # back propagation
            optimizer.step()            # step the gradients once
            optimizer.zero_grad()       # clear the gradients before the next step
            loss1 = stance_loss.item()  # get the value of the loss
            loss_values.append(loss1)   # archive the loss
            if len(loss_horz)==0:
                loss_horz.append(0)
            else:
                loss_horz.append(len(loss_horz))
        
        mappinglayer.eval() # change the layer back to eval mode
        
        # after 1 epoch, measure the f1 score on eval set
        results = test(model=model, 
                       mappinglayer=mappinglayer, 
                       dataloader=eval_dl,
                       log_interval=log_interval,
                       num_post=num_post, 
                       post_length=post_length)
        stance_pred = results[0]    # shape is (NA,)
        stance_true = results[1]    # shape is (NA,)
        stance_pred = stance_pred.to('cpu')
        stance_true = stance_true.to('cpu')
        # select only the replies to do analysis on. SRQ dataset only has parent, 1 reply. ignore all other labels
        start=1
        stop = stance_pred.shape[0]
        step = num_post
        index_2_pick = range(start,stop,step)
        
        stance_pred = stance_pred[index_2_pick]
        stance_true = stance_true[index_2_pick]
        stance_metrics = helper.stance_f1(stance_pred,          # calculate the f1-metrics for stance
                                          stance_true,
                                          incl_empty=False)
        
        stance_accuracy = helper.accuracy_stance(stance_pred,   # calculate prediction accuracies
                                                 stance_true)
        
        stance_msg = helper.stance_f1_msg(stance_metrics[0],    # Get the strings to display for f1 scores
                                          stance_metrics[1],
                                          stance_metrics[2],
                                          stance_metrics[3],
                                          stance_metrics[4],
                                          incl_empty=False)
        logger.info('\n'+stance_msg)
        logger.info('Stance Accuracy: %1.4f' % stance_accuracy)
        
        f1_score = stance_metrics[-1]
        f1_scores.append(f1_score)
        f1_horz.append(epoch)
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(mappinglayer.state_dict(), filename)  # save the trained mapping
    
    weights_list.append(mappinglayer.linear1.weight)
    torch.save(weights_list, './log_files/weights_backup.pkl')
    
    # reload best mapping function
    state = torch.load(filename)
    mappinglayer.load_state_dict(state)
    fig, axes = plt.subplots(2,1)
    ax0 = axes[0] 
    ax0.plot(loss_horz, loss_values)
    ax0.grid(True)
    ax1 = axes[1]
    ax1.plot(f1_horz, f1_scores)
    ax1.grid(True)
    fig.savefig(plotfile)
    return


def test(model, mappinglayer, dataloader, log_interval, num_post=4, post_length=128):
    logger = logging.getLogger(__name__)
    
    model.eval() # set model to test mode first
    mappinglayer.eval()
    
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    
    # Start the arrays to store entire dataset
    premap_logits_arr = None
    mapped_logits_arr = None
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
            #length_labels = minibatch[4]
            stance_labels = minibatch[5]
            
            # if necessary, extend data to 7 post long if model expects more data
            if num_post > 4:
                mb = encoded_comments.shape[0]
                encoded_comments_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                encoded_comments_2[:,:post_length*4] = encoded_comments[:,:post_length*4] # extend to (n, however needed)
                encoded_comments = encoded_comments_2
                token_type_ids_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                token_type_ids_2[:,:post_length*4] = token_type_ids[:,:post_length*4] # extend to (n, however needed)
                token_type_ids = token_type_ids_2
                attention_masks_2 = torch.zeros((mb, post_length * num_post), dtype=torch.long)
                attention_masks_2[:,:post_length*4] = attention_masks[:,:post_length*4] # extend to (n, however needed)
                attention_masks = attention_masks_2
                
                stance_labels_2 = torch.ones((mb, num_post), dtype=torch.long) * -1
                stance_labels_2[:, :4] = stance_labels[:, :4]
                stance_labels = stance_labels_2
            
            # move features to gpu
            encoded_comments = encoded_comments.to(gpu)
            token_type_ids = token_type_ids.to(gpu)
            attention_masks = attention_masks.to(gpu)
            # length_labels = length_labels.to(gpu)
            # stance_labels = stance_labels.to(gpu)
            
            # get the stance prediction logits
            stance_logits = model(input_ids = encoded_comments,         # shape (n,A,B) where 
                                  token_type_ids = token_type_ids,      # n: minibatch
                                  attention_masks = attention_masks,    # A: num posts per thread
                                  task='stance')                        # B: num of stance classes 
            mapped_logits = mappinglayer(stance_logits)
            # no need to test length cauz SRQ lengths are all 2
            
            if mapped_logits_arr is None:
                mapped_logits_arr = mapped_logits.to(cpu)               # for handling first minibatch only
                premap_logits_arr = stance_logits.to(cpu)
                stance_labels_arr = stance_labels.long()
            else:
                mapped_logits_arr = torch.cat((mapped_logits_arr,       # shape (N,A,B)
                                               mapped_logits.to(cpu)),  # N is entire dataset length
                                              0)
                premap_logits_arr = torch.cat((premap_logits_arr,       # shape (N,A,B)
                                               stance_logits.to(cpu)),  # N is entire dataset length
                                              0)
                stance_labels_arr = torch.cat((stance_labels_arr,       # shape is (NA,1)
                                               stance_labels.long()),
                                              0)
        
        stance_pred =helper.logit_2_class_stance(mapped_logits_arr) # convert logits to stance labels. (NA,)
        stance_true = stance_labels_arr.reshape(stance_pred.shape)  # reshape from (N,A) into (NA,)
        premapped = helper.logit_2_class_stance(premap_logits_arr)  # convert logits to stance labels. (NA,)
    return [stance_pred, stance_true, premapped]


def get_model(modeldir, modelfile, 
              max_post_num=4,
              max_post_len=256):
    ''' Returns the model, with weights loaded '''
    logger = logging.getLogger(__name__)
    if 'alt_modelf'==modelfile.lower()[0:10]:
        number = int(modelfile[10])
        model = alt_ModelFn.from_pretrained('bert-base-uncased',
                                            max_post_length=max_post_len,
                                            num_transformers=number)
    else:
        logger.info('Exiting, model not found: ' + modelfile)
        raise Exception
    
    model.resize_token_embeddings(len(DataProcessor.default_tokenizer))
    model = model.cuda()                    # put the model into GPU
    model = torch.nn.DataParallel(model)    # wrap the model using DataParallel
    
    temp = torch.load(modeldir+modelfile)   # load the best model
    model.load_state_dict(temp)         # stuff state into model
    model = model.module                # get back the non parallel model
    return model

def get_addon(addondir, addonfile):
    ''' Returns the last layer that does mapping from 11 classes to 5 classes'''
    logger = logging.getLogger(__name__)
    addon = final_mapping_layer()
    try:
        temp = torch.load(addondir+addonfile)  # load the best model
        logger.info('Final mapping found: ' + addonfile)
        addon.load_state_dict(temp)
    except Exception:
        logger.info('Final mapping layer not found: '+ addonfile+'. Using random init mapping scheme model. ')
    return addon

def map_coarse_discourse_2_sdqc_labels(input_tensor, option=1):
    '''
    maps tensor of labels from coarse disc 11 types to SDQC 5 types
    mapping table from coarse_discourse into semeval17 format
    =========== option 1 ===========
    Reddit + Empty ==> SDQC + Empty
    0.empty        ==> 0.empty
    1.question     ==> 3.query
    2.answer       ==> 4.comment
    3.announcement ==> 4.comment
    4.agreement    ==> 2.support
    5.appreciation ==> 4.comment
    6.disagreement ==> 1.deny
    7.-ve reaction ==> 1.deny
    8.elaboration  ==> 4.comment
    9.humor        ==> 4.comment
    10.other       ==> 4.comment
    
    =========== option 2 ===========
    change elaboration to support, appreciation to support
    Reddit + Empty ==> SDQC + Empty
    0.empty        ==> 0.empty
    1.question     ==> 3.query
    2.answer       ==> 4.comment
    3.announcement ==> 4.comment
    4.agreement    ==> 2.support
    5.appreciation ==> 2.support
    6.disagreement ==> 1.deny
    7.-ve reaction ==> 1.deny
    8.elaboration  ==> 2.support
    9.humor        ==> 4.comment
    10.other       ==> 4.comment
    
    '''
    output_tensor = torch.zeros(size=input_tensor.size())
    if option==1:
        for i in range(len(input_tensor)):
            value = input_tensor[i]
            if (value==0):
                output_tensor[i] = 0
            elif (value==6) or (value==7):
                output_tensor[i] = 1
            elif (value==4) :
                output_tensor[i] = 2
            elif (value==1):
                output_tensor[i] = 3
            else:
                output_tensor[i] = 4
    elif option==2:
        for i in range(len(input_tensor)):
            value = input_tensor[i]
            if (value==0):
                output_tensor[i] = 0
            elif (value==6) or (value==7):
                output_tensor[i] = 1
            elif (value==4) or (value==5) or (value==8):
                output_tensor[i] = 2
            elif (value==1):
                output_tensor[i] = 3
            else:
                output_tensor[i] = 4
            
    return output_tensor

def plot_weights():
    weights_list = torch.load('./log_files/weights_backup.pkl')
    datalen = len(weights_list)
    
    arr1 = np.zeros((datalen, 5, 11))
    for i in range(datalen):
        arr1[i,:,:] = weights_list[i].cpu().detach().numpy()
    
    fig, ax = plt.subplots(4,10, sharex=True)
    for i in range(1,5):
        for j in range(1,11):
            axis = ax[i-1][j-1]
            axis.plot(arr1[:,i,j])
            axis.grid(True)

            
if __name__ == '__main__':
    ans = main()
    
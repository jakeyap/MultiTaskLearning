#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:57:59 2021
The models in this file are written to avoid the huggingface 
self attention implementation in explicit transformer blocks
@author: jakeyap
"""

import torch
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel, BertPooler, BertLayer
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from classifier_models import BertHierarchyPooler # for pulling out the [CLS] embeddings

# TODO create a model that scans the appropriate kids for length task

class alt_ModelEn(BertPreTrainedModel):
    # For the Coarse Discourse original 10 stance
    # exposed posts means how many to check for length prediction
    def __init__(self, config, stance_num_labels=11, length_num_labels=2, max_post_num=10, exposed_posts=4, max_post_length=256,num_transformers=1):
        super(alt_ModelEn, self).__init__(config)
        self.length_num_labels = length_num_labels
        self.stance_num_labels = stance_num_labels
        self.max_post_num = max_post_num
        self.exposed_posts = exposed_posts
        self.max_post_length = max_post_length
        self.num_transformers = num_transformers

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        # a single self attention layer
        single_tf_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, 
                                                     nhead=8, 
                                                     dropout=config.hidden_dropout_prob)
        
        # create transformer blocks. the single layer is cloned internally
        self.transformer_stance = nn.TransformerEncoder(single_tf_layer,
                                                        num_layers=num_transformers)
        
        self.transformer_length = nn.TransformerEncoder(single_tf_layer,
                                                        num_layers=num_transformers)
        
        self.max_post_num = max_post_num
        self.max_post_length = max_post_length
        self.pooler = BertHierarchyPooler(config)
        
        self.length_classifier = nn.Linear(exposed_posts * config.hidden_size, length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, stance_num_labels)
        # for initializing all weights
        self.apply(self._init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, task):
        # input_ids, token_type_ids, attention_masks dimensions are all (n, AxB),
        # where n=minibatch_size, A=max_post_length, B=max_post_num. B is hard coded to 4 in this model
        idx = self.max_post_length
        
        # Reshape the vectors, then pass into BERT. 
        # Each sequence_outputn is a post. Dimension is (n, A, hidden_size). n=minibatch_size, A=max_post_length
        # Stick them together to become (n, AxB, hidden_size)
        sequence_output0, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output = sequence_output0
        for i in range(1, self.max_post_num):
            sequence_outputn, _ = self.bert(input_ids[:,i*idx:(i+1)*idx], 
                                            token_type_ids[:,i*idx:(i+1)*idx], 
                                            attention_masks[:,i*idx:(i+1)*idx])
            sequence_output = torch.cat((sequence_output, sequence_outputn), dim=1)
        
        # the attention mask size must be (n, 1, 1, length) where n is minibatch
        attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
        
        ''' Try just pooling the CLS labels '''
        cls_positions = self.pooler(sequence_output,        # shape=(n,B,num_hiddens) where n=minibatch, B=max_post_num
                                    self.max_post_length, 
                                    self.max_post_num)
        
        stance_logits = None
        length_logits = None
        
        if (task=='length'):                    # for length prediction task
            num_posts = self.exposed_posts                  # just look at the first few posts to predict length
            hiddens = cls_positions[:, 0:num_posts, :]      # shape = (n,C,num_hiddens) where C=exposed_posts
            hiddens = self.transformer_length(hiddens)      # shape = (n,C,num_hiddens)
            
            mb_size = hiddens.shape[0]                      # find n, minibatch size
            hiddens = hiddens.reshape(mb_size,-1)           # reshape before passing into neural net (n, C x num_hidden)
            length_logits = self.length_classifier(hiddens) # (n, num of length classes)
            return length_logits
        
        elif (task=='stance'):                  # for stance classification task
            hiddens =self.transformer_stance(cls_positions) # shape = (n,B,num_hiddens) where B=max_post_num
            stance_logits = self.stance_classifier(hiddens) # (n, B, num of stance classes)
            return stance_logits
        else:
            print('task is ' +task+ '. Must be "stance", "length".')
            raise Exception
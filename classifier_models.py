#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:08:03 2020

@author: jakeyap
"""

import copy
import torch
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel, BertPooler, BertLayer
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn as nn

class BertHierarchyPooler(nn.Module):
    def __init__(self, config):
        super(BertHierarchyPooler, self).__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation = nn.Tanh()

    def forward(self, hidden_states, max_post_length, max_post_num):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0].unsqueeze(1)
        
        # for extracting encoded CLS tokens from each of the encoded posts
        for i in range(1, max_post_num):
            tmp_token_tensor = hidden_states[:, max_post_length * i].unsqueeze(1)
            if i == 1:
                tmp_output = torch.cat((first_token_tensor, tmp_token_tensor), dim=1)
            else:
                tmp_output = torch.cat((tmp_output, tmp_token_tensor), dim=1)

        final_output = tmp_output
        return final_output

    
class my_ModelA0(BertPreTrainedModel):
    def __init__(self, config, stance_num_labels=5, length_num_labels=2, max_post_num=4, max_post_length=256):
        super(my_ModelA0, self).__init__(config)
        self.length_num_labels = length_num_labels
        self.stance_num_labels = stance_num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.add_length_bert_attention = BertEncoderOneLayer(config)
        self.add_stance_bert_attention = BertEncoderOneLayer(config)
        self.max_post_num = max_post_num
        self.max_post_length = max_post_length
        self.stance_pooler = BertHierarchyPooler(config)
        self.length_pooler = BertHierarchyPooler(config)
        self.length_classifier = nn.Linear(max_post_num * config.hidden_size, length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, stance_num_labels)
        # for initializing all weights
        self.apply(self._init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, 
                task, stance_labels=None):
        # input_ids, token_type_ids, attention_masks dimensions are all (n, AxB),
        # where n=minibatch_size, A=max_post_length, B=max_post_num. B is hard coded to 4 in this model
        idx = self.max_post_length
        
        # Reshape the vectors, then pass into BERT. 
        # Each sequence_output is a post. Dimension is (n, A, hidden_size). n=minibatch_size, A=max_post_length
        sequence_output1, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output2, _ = self.bert(input_ids[:,1*idx:2*idx], token_type_ids[:,1*idx:2*idx], attention_masks[:,1*idx:2*idx])
        sequence_output3, _ = self.bert(input_ids[:,2*idx:3*idx], token_type_ids[:,2*idx:3*idx], attention_masks[:,2*idx:3*idx])
        sequence_output4, _ = self.bert(input_ids[:,3*idx:4*idx], token_type_ids[:,3*idx:4*idx], attention_masks[:,3*idx:4*idx])
        
        # Stick the outputs back together. 
        # sequence_output dimension = (n, AxB, hidden_size). n=minibatch_size, A=max_post_length, B=max_post_num
        tmp_sequence = torch.cat((sequence_output1, sequence_output2), dim=1)
        tmp_sequence = torch.cat((tmp_sequence, sequence_output3), dim=1)
        sequence_output = torch.cat((tmp_sequence, sequence_output4), dim=1)
        
        if task == 'length':    #for length prediction task
            # the attention mask size must be (n, 1, 1, length) where n is minibatch
            attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
            add_length_bert_encoder = self.add_length_bert_attention(sequence_output, attention_masks)
            
            # take the last hidden layer as output
            final_length_text_output = add_length_bert_encoder[-1]
            # data is formatted as a tensor inside a tuple
            final_length_text_output = final_length_text_output [0]
            pooled_output_length = self.length_pooler(final_length_text_output,
                                                      self.max_post_length, 
                                                      self.max_post_num)

            length_output = self.dropout(pooled_output_length)      # shape=(n,A,num_hiddens) where n=minibatch, A=max_post_length
            minibatch_size = length_output.shape[0]
            length_output = length_output.reshape(minibatch_size, -1)
            length_logits = self.length_classifier(length_output)   # shape=(n,2) where n=minibatch size
            return length_logits

        elif task == 'stance':  # for stance classification task
            # the attention mask size must be (n, 1, 1, length) where n is minibatch
            attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
            add_stance_bert_encoder = self.add_stance_bert_attention(sequence_output, 
                                                                     attention_masks)
            
            # take the last hidden layer as output
            final_stance_text_output = add_stance_bert_encoder[-1]
            # data is formatted as a tensor inside a tuple
            final_stance_text_output = final_stance_text_output [0]
            pooled_output_stance = self.stance_pooler(final_stance_text_output, 
                                                      self.max_post_length, 
                                                      self.max_post_num)
            stance_output = self.dropout(pooled_output_stance)
            stance_logits = self.stance_classifier(stance_output)
            
            if stance_labels is not None:
                loss_fct = CrossEntropyLoss()
                # Only keep active parts of the loss
                num_labels = stance_labels.size()[0]
                active_logits = stance_logits[0:num_labels]
                loss = loss_fct(active_logits, stance_labels)
                return loss
            else:
                return stance_logits
        else:
            print('task must be "stance" or "length"')
            raise Exception

class BertEncoderOneLayer(torch.nn.Module):
    # copied from BertEncoder class in transformers pkg
    # the only difference is this is 1 layer ONLY, the original is 12
    def __init__(self, config):
        super(BertEncoderOneLayer, self).__init__()
        layer = BertLayer(config)
        self.layer = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
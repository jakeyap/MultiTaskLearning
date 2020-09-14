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
   
class BertStancePooler(nn.Module):
    def __init__(self, config):
        super(BertStancePooler, self).__init__()
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.activation = nn.Tanh()

    def forward(self, hidden_states, max_tweet_num, max_tweet_len):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        max_bucket_num = 5
        max_seq_len = 512

        first_token_tensor = hidden_states[:, 0].unsqueeze(1)
        
        # for extracting encoded CLS tokens from the 1st bucket
        for i in range(1, max_tweet_num):
            tmp_token_tensor = hidden_states[:, max_tweet_len * i].unsqueeze(1)
            if i == 1:
                tmp_output = torch.cat((first_token_tensor, tmp_token_tensor), dim=1)
            else:
                tmp_output = torch.cat((tmp_output, tmp_token_tensor), dim=1)
        
        # for extracting encoded CLS tokens from the remaining buckets
        for j in range(1, max_bucket_num):
            for k in range(max_tweet_num):
                tmp_token_tensor = hidden_states[:, max_seq_len * j + max_tweet_len * k].unsqueeze(1)
                tmp_output = torch.cat((tmp_output, tmp_token_tensor), dim=1)

        final_output = tmp_output
        return final_output

    
class my_ModelA0(BertPreTrainedModel):
    def __init__(self, config, stance_num_labels=4, length_num_labels=2, max_post_num=4, max_post_length=64):
        super(my_ModelA0, self).__init__(config)
        self.length_num_labels = length_num_labels
        self.stance_num_labels = stance_num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.length_pooler = BertPooler(config)
        self.classifier0 = torch.nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.add_length_bert_attention = BertEncoderOneLayer(config)
        self.add_stance_bert_attention = BertEncoderOneLayer(config)
        self.max_post_num = max_post_num
        self.max_post_length = max_post_length
        self.stance_pooler = BertStancePooler(config)
        self.length_classifier = nn.Linear(config.hidden_size, length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, stance_num_labels)
        # for initializing all weights
        self.apply(self.init_bert_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, 
                attention_mask, task=None, stance_label_mask=None):

        sequence_output1, _ = self.bert(input_ids[1], token_type_ids[1], attention_masks[1], output_all_encoded_layers=False)
        sequence_output2, _ = self.bert(input_ids[2], token_type_ids[2], attention_masks[2], output_all_encoded_layers=False)
        sequence_output3, _ = self.bert(input_ids[3], token_type_ids[3], attention_masks[3], output_all_encoded_layers=False)
        sequence_output4, _ = self.bert(input_ids[4], token_type_ids[4], attention_masks[4], output_all_encoded_layers=False)
        
        attention_mask = []
        for eachmask in attention_masks:
            attention_mask.extend(eachmask)
        attention_mask = torch.tensor(attention_mask)
        
        tmp_sequence = torch.cat((sequence_output1, sequence_output2), dim=1)
        tmp_sequence = torch.cat((tmp_sequence, sequence_output3), dim=1)
        sequence_output = torch.cat((tmp_sequence, sequence_output4), dim=1)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if task is None: #for length prediction task
            add_length_bert_encoder = self.add_length_bert_attention(sequence_output, extended_attention_mask)
            add_length_bert_text_output_layer = add_length_bert_encoder[-1]
            final_length_text_output = self.length_pooler(add_length_bert_text_output_layer)

            length_pooled_output = self.dropout(final_length_text_output)
            logits = self.length_classifier(length_pooled_output)
            return logits
        else:
            # for stance classification task
            add_stance_bert_encoder = self.add_stance_bert_attention(sequence_output, extended_attention_mask)
            final_stance_text_output = add_stance_bert_encoder[-1]

            label_logit_output = self.stance_pooler(final_stance_text_output)
            sequence_stance_output = self.dropout(label_logit_output)
            stance_logits = self.stance_classifier(sequence_stance_output)
            return stance_logits

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
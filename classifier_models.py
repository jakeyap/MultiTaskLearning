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
from torch.nn import CrossEntropyLoss
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

        final_output = tmp_output  # size: (n,A,hidden_size) where n=minibatch, A=max_post_num
        return final_output

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


class BertEncoderCustom(nn.Module):
    ''' 
    same as the transformers library BERTEncoder, but dumbed down by YK
    '''
    def __init__(self, config, num_transformers):
        super(BertEncoderCustom, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_transformers)])

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return (all_hidden_states, attention_mask)


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
        # sequence_output shape = (n, AxB, hidden_size). n=minibatch_size, A=max_post_length, B=max_post_num
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
            final_length_text_output = final_length_text_output [0]             # shape=(n,AxB,hidden_size)
            
            pooled_output_length = self.length_pooler(final_length_text_output, # shape=(n,B,hidden_size)
                                                      self.max_post_length, 
                                                      self.max_post_num)

            length_output = self.dropout(pooled_output_length)      # shape=(n,B,hidden_size) where n=minibatch, B=max_post_num
            batch_size = length_output.shape[0]
            length_output = length_output.reshape(batch_size ,-1)   # shape=(n,A x hidden_size)
            length_logits = self.length_classifier(length_output)   # shape=(n,length_num_labels)
            return length_logits

        elif task == 'stance':  # for stance classification task
            # the attention mask size must be (n, 1, 1, length) where n is minibatch
            attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
            add_stance_bert_encoder = self.add_stance_bert_attention(sequence_output, 
                                                                     attention_masks)
            
            # take the last hidden layer as output
            final_stance_text_output = add_stance_bert_encoder[-1]
            # data is formatted as a tensor inside a tuple
            final_stance_text_output = final_stance_text_output [0]             # shape=(n,AxB,hidden_size)
            pooled_output_stance = self.stance_pooler(final_stance_text_output, # shape=(n,B,hidden_size)
                                                      self.max_post_length, 
                                                      self.max_post_num)
            stance_output = self.dropout(pooled_output_stance)      # shape=(n,B,hidden_size)
            stance_logits = self.stance_classifier(stance_output)   # shape=(n,B,stance_num_labels)
            
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

class my_ModelBn(BertPreTrainedModel):
    def __init__(self, config, stance_num_labels=5, length_num_labels=2, max_post_num=4, max_post_length=256,num_transformers=1):
        super(my_ModelBn, self).__init__(config)
        self.length_num_labels = length_num_labels
        self.stance_num_labels = stance_num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.transformer_layer = BertEncoderCustom(config,num_transformers)
        
        self.max_post_num = max_post_num
        self.max_post_length = max_post_length
        self.pooler = BertHierarchyPooler(config)
        
        self.length_classifier = nn.Linear(max_post_num * config.hidden_size, length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, stance_num_labels)
        # for initializing all weights
        self.apply(self._init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, task):
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
        # the attention mask size must be (n, 1, 1, length) where n is minibatch
        attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
            
        if task == 'length':    #for length prediction task
            text_output = self.transformer_layer(sequence_output, attention_masks)
            text_output = text_output[0]                # data is a tensor inside a tuple
            text_output = text_output[-1]               # take the last hidden layer as output. shape = (n, B, hidden_size)
            
            pooled_output = self.pooler(text_output,    # shape=(n,B,num_hiddens) where n=minibatch, B=max_post_num
                                        self.max_post_length, 
                                        self.max_post_num)

            length_output = self.dropout(pooled_output) # shape=(n,B,num_hiddens)
            batch_size = length_output.shape[0]
            length_output = length_output.reshape(batch_size, -1)
            length_logits = self.length_classifier(length_output)   # shape=(n,2) where n=minibatch size
            return length_logits

        elif task == 'stance':  # for stance classification task
            text_output = self.transformer_layer(sequence_output, attention_masks)
            text_output = text_output[0]                # data is a tensor inside a tuple
            text_output = text_output[-1]               # take the last hidden layer as output. shape = (n, B, hidden_size)
            pooled_output = self.pooler(text_output,    # shape=(n,B,num_hiddens) where n=minibatch, B=max_post_num
                                        self.max_post_length, 
                                        self.max_post_num)
            
            stance_output = self.dropout(pooled_output)
            stance_logits = self.stance_classifier(stance_output)
            return stance_logits
        else:
            print('task must be "stance" or "length"')
            raise Exception    

# ModelCn is skipped. It is the model with dual LR for stance/length tasks

class my_ModelDn(BertPreTrainedModel):
    # For the Coarse Discourse original 10 stance
    # exposed posts means how many to check for length prediction
    def __init__(self, config, stance_num_labels=11, length_num_labels=2, max_post_num=10, exposed_posts=4, max_post_length=256,num_transformers=1):
        super(my_ModelDn, self).__init__(config)
        self.length_num_labels = length_num_labels
        self.stance_num_labels = stance_num_labels
        self.max_post_num = max_post_num
        self.exposed_posts = exposed_posts
        self.max_post_length = max_post_length
        self.num_transformers = num_transformers

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.transformer_layer = BertEncoderCustom(config,num_transformers)
        
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
        ''' # Legacy hard coded style from ModelA0-Bn
        sequence_output1, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output2, _ = self.bert(input_ids[:,1*idx:2*idx], token_type_ids[:,1*idx:2*idx], attention_masks[:,1*idx:2*idx])
        sequence_output3, _ = self.bert(input_ids[:,2*idx:3*idx], token_type_ids[:,2*idx:3*idx], attention_masks[:,2*idx:3*idx])
        sequence_output4, _ = self.bert(input_ids[:,3*idx:4*idx], token_type_ids[:,3*idx:4*idx], attention_masks[:,3*idx:4*idx])
        
        # Stick the outputs back together. 
        # sequence_output dimension = (n, AxB, hidden_size). n=minibatch_size, A=max_post_length, B=max_post_num
        tmp_sequence = torch.cat((sequence_output1, sequence_output2), dim=1)
        tmp_sequence = torch.cat((tmp_sequence, sequence_output3), dim=1)
        sequence_output = torch.cat((tmp_sequence, sequence_output4), dim=1)
        '''        
        # the attention mask size must be (n, 1, 1, length) where n is minibatch
        attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
        
        ''' Try just pooling the CLS labels '''
        cls_positions = self.pooler(sequence_output,        # shape=(n,B,num_hiddens) where n=minibatch, B=max_post_num
                                    self.max_post_length, 
                                    self.max_post_num)
        
        if (task=='length'):                # for length prediction task
            num_posts = self.exposed_posts                  # just look at the first few posts to predict length
            hiddens = cls_positions[:, 0:num_posts, :]      # shape = (n,C,num_hiddens) where C=exposed_posts
            mb_size = hiddens.shape[0]                      # find n, minibatch size
            hiddens = hiddens.reshape(mb_size,-1)           # reshape before passing into neural net (n, C x num_hidden)
            length_logits = self.length_classifier(hiddens) # (n, num of length classes)
            return length_logits

        elif (task=='stance'):              # for stance classification task
            hiddens = self.transformer_layer(cls_positions) # tuple of length 1
            hiddens = hiddens[0]                            # get the list of attention layers outputs. need the last one 
            hiddens = hiddens[-1]                           # shape = (n,B,num_hiddens) where B=max_post_num
            stance_logits = self.stance_classifier(hiddens) # (n, B, num of stance classes)
            return stance_logits
        else:
            print('task is ' +task+ '. Must be "stance" or "length".')
            raise Exception
            
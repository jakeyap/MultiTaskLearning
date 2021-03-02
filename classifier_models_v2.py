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

class final_mapping_layer(nn.Module):
    def __init__(self,init=True):
        super(final_mapping_layer, self).__init__()
        self.linear1 = nn.Linear(11,5)
        #self.linear2 = nn.Linear(5,5)
        if init:
            with torch.no_grad():
                weight = self.linear1.weight
                bias = self.linear1.bias
                weight[0,0] += 10
                weight[3,1] += 10
                weight[4,2] += 10
                weight[4,3] += 10
                weight[2,4] += 10
                weight[2,5] += 10
                weight[1,6] += 10
                weight[1,7] += 10
                weight[2,8] += 10
                weight[4,9] += 10
                weight[4,10] += 10
                weight = weight / 10
                self.linear1.weight[:,:] = weight[:,:]
                bias = bias / 10
                self.linear1.bias[:] = bias[:]
    
    def forward(self,x):
        x = self.linear1(x)
        # x = torch.sigmoid(x)
        #x = self.linear2(x)
        return x
    

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
            print('task is "' +task+ '". Must be "stance", "length".')
            raise Exception
            
class alt_ModelFn(BertPreTrainedModel):
    # For the Coarse Discourse original 10 stance
    # exposed posts means how many to check for length prediction
    # for handling strategy 1, (1 root + 3 child + 3 grandkids) for stance prediction
    # check 1 root + 3 child for length prediction
    
    def __init__(self, config, max_post_length=256, num_transformers=1):
        super(alt_ModelFn, self).__init__(config)
        self.length_num_labels = 2
        self.stance_num_labels = 11 # 10 class + isempty
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
        
        self.max_post_length = max_post_length
        self.pooler = BertHierarchyPooler(config)
        
        self.length_classifier = nn.Linear(4 * config.hidden_size, self.length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, self.stance_num_labels)
        # for initializing all weights
        self.apply(self._init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, task):
        # input_ids, token_type_ids, attention_masks dimensions are all (n, A x 7),
        # where n=minibatch_size, A=max_post_length
        idx = self.max_post_length
        
        # Reshape the vectors, then pass into BERT. 
        # Each sequence_outputn is a post. Dimension is (n, A, hidden_size). n=minibatch_size, A=max_post_length
        # Stick them together to become (n, Ax7, hidden_size)
        sequence_output0, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output = sequence_output0
        for i in range(1, 7):
            sequence_outputn, _ = self.bert(input_ids[:,i*idx:(i+1)*idx], 
                                            token_type_ids[:,i*idx:(i+1)*idx], 
                                            attention_masks[:,i*idx:(i+1)*idx])
            sequence_output = torch.cat((sequence_output, sequence_outputn), dim=1)
        
        # the attention mask size must be (n, 1, 1, length) where n is minibatch
        # attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
        
        ''' Try just pooling the CLS labels '''
        cls_positions = self.pooler(sequence_output,        # shape=(n,7,num_hiddens) where n=minibatch
                                    self.max_post_length, 
                                    7)
        
        # apply dropout. forgot about this previously. 2021 jan 14
        cls_positions = self.dropout(cls_positions)
        
        stance_logits = None
        length_logits = None
        
        if (task=='length'):                    # for length prediction task
            hiddens = cls_positions[:, (0,1,3,5), :]        # shape = (n,4,num_hiddens) pick the root and kids only
            hiddens = self.transformer_length(hiddens)      # shape = (n,4,num_hiddens)
            
            mb_size = hiddens.shape[0]                      # find n, minibatch size
            hiddens = hiddens.reshape(mb_size,-1)           # reshape before passing into neural net (n, 4 x num_hidden)
            length_logits = self.length_classifier(hiddens) # (n, num of length classes)
            return length_logits
        
        elif (task=='stance'):                  # for stance classification task
            hiddens =self.transformer_stance(cls_positions) # shape = (n,7,num_hiddens)
            stance_logits = self.stance_classifier(hiddens) # (n, 7, num of stance classes)
            return stance_logits
        else:
            print('task is "' +task+ '". Must be "stance", "length".')
            raise Exception
            
class alt_ModelGn(BertPreTrainedModel):
    # For the Coarse Discourse original 10 stance
    # exposed posts means how many to check for length prediction
    # for handling strategy 1, (1 root + 3 child + 3 grandkids) for stance prediction
    # check 1 root + 3 child for length prediction
    
    # instead of taking the CLS token only, use maxpool of all encoded tokens
    
    def __init__(self, config, max_post_length=256, num_transformers=1):
        super(alt_ModelGn, self).__init__(config)
        self.length_num_labels = 2
        self.stance_num_labels = 11 # 10 class + isempty
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
        
        self.max_post_length = max_post_length
        
        # self.pooler = BertHierarchyPooler(config)
        self.pooler = nn.MaxPool1d(max_post_length)
        
        self.length_classifier = nn.Linear(4 * config.hidden_size, self.length_num_labels)
        self.stance_classifier = nn.Linear(config.hidden_size, self.stance_num_labels)
        # for initializing all weights
        self.apply(self._init_weights)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks, task):
        # input_ids, token_type_ids, attention_masks dimensions are all (n, A x 7),
        # where n=minibatch_size, A=max_post_length
        idx = self.max_post_length
        
        # Reshape the vectors, then pass into BERT. 
        # Each sequence_outputn is a post. Dimension is (n, A, hidden_size). n=minibatch_size, A=max_post_length
        # Stick them together to become (n, Ax7, hidden_size)
        '''
        sequence_output0, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output = sequence_output0
        for i in range(1, 7):
            sequence_outputn, _ = self.bert(input_ids[:,i*idx:(i+1)*idx], 
                                            token_type_ids[:,i*idx:(i+1)*idx], 
                                            attention_masks[:,i*idx:(i+1)*idx])
            sequence_output = torch.cat((sequence_output, sequence_outputn), dim=1)
        '''
        sequence_output0, _ = self.bert(input_ids[:,0*idx:1*idx], token_type_ids[:,0*idx:1*idx], attention_masks[:,0*idx:1*idx])
        sequence_output1, _ = self.bert(input_ids[:,1*idx:2*idx], token_type_ids[:,1*idx:2*idx], attention_masks[:,1*idx:2*idx])
        sequence_output2, _ = self.bert(input_ids[:,2*idx:3*idx], token_type_ids[:,2*idx:3*idx], attention_masks[:,2*idx:3*idx])
        sequence_output3, _ = self.bert(input_ids[:,3*idx:4*idx], token_type_ids[:,3*idx:4*idx], attention_masks[:,3*idx:4*idx])
        sequence_output4, _ = self.bert(input_ids[:,4*idx:5*idx], token_type_ids[:,4*idx:5*idx], attention_masks[:,4*idx:5*idx])
        sequence_output5, _ = self.bert(input_ids[:,5*idx:6*idx], token_type_ids[:,5*idx:6*idx], attention_masks[:,5*idx:6*idx])
        sequence_output6, _ = self.bert(input_ids[:,6*idx:7*idx], token_type_ids[:,6*idx:7*idx], attention_masks[:,6*idx:7*idx])
        
        # the attention mask size must be (n, 1, 1, length) where n is minibatch
        # attention_masks = attention_masks.unsqueeze(1).unsqueeze(2)
        post_length = self.max_post_length
        hidden_size = sequence_output0.shape[-1]
        mb_size = sequence_output0.shape[0]
        
        # reshape each post into new shape [n, 768, max_post_len]
        sequence_output0 = sequence_output0.reshape([mb_size, hidden_size, post_length])
        sequence_output1 = sequence_output1.reshape([mb_size, hidden_size, post_length])
        sequence_output2 = sequence_output2.reshape([mb_size, hidden_size, post_length])
        sequence_output3 = sequence_output3.reshape([mb_size, hidden_size, post_length])
        sequence_output4 = sequence_output4.reshape([mb_size, hidden_size, post_length])
        sequence_output5 = sequence_output5.reshape([mb_size, hidden_size, post_length])
        sequence_output6 = sequence_output6.reshape([mb_size, hidden_size, post_length])
        
        # after pooling, shape is now [n, 768, 1]
        pooled0 = self.pooler(sequence_output0)
        pooled1 = self.pooler(sequence_output1)
        pooled2 = self.pooler(sequence_output2)
        pooled3 = self.pooler(sequence_output3)
        pooled4 = self.pooler(sequence_output4)
        pooled5 = self.pooler(sequence_output5)
        pooled6 = self.pooler(sequence_output6)
        
        ''' Try just pooling the CLS labels '''
        '''
        cls_positions = self.pooler(sequence_output,        # shape=(n,7,num_hiddens) where n=minibatch
                                    self.max_post_length, 
                                    7)'''
        # shape is (n, 7 x max_post_len, hidden_size)
        
        
        pooled_outputs = torch.cat((pooled0, pooled1,   # shape is now (n, hidden_size, 7)
                                    pooled2, pooled3,
                                    pooled4, pooled5, 
                                    pooled6), dim=2) 
        
        # apply dropout. forgot about this previously. 2021 jan 14
        pooled_outputs = self.dropout(pooled_outputs)
        
        pooled_outputs = pooled_outputs.reshape((mb_size, 7, hidden_size))  # reshape into (n,7,num_hiddens)
        cls_positions = pooled_outputs
        
        stance_logits = None
        length_logits = None
        
        if (task=='length'):                    # for length prediction task
            hiddens = cls_positions[:, (0,1,3,5), :]        # shape = (n,4,num_hiddens) pick the root and kids only
            hiddens = self.transformer_length(hiddens)      # shape = (n,4,num_hiddens)
            
            mb_size = hiddens.shape[0]                      # find n, minibatch size
            hiddens = hiddens.reshape(mb_size,-1)           # reshape before passing into neural net (n, 4 x num_hidden)
            length_logits = self.length_classifier(hiddens) # (n, num of length classes)
            return length_logits
        
        elif (task=='stance'):                  # for stance classification task
            hiddens =self.transformer_stance(cls_positions) # shape = (n,7,num_hiddens)
            stance_logits = self.stance_classifier(hiddens) # (n, 7, num of stance classes)
            return stance_logits
        else:
            print('task is "' +task+ '". Must be "stance", "length".')
            raise Exception
        
class SelfAdjDiceLoss(torch.nn.Module):
    """
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:27:49 2020

@author: jakeyap
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from preprocessor_functions import remove_urls, remove_spaces, post_isempty
from convert_coarse_discourse import extract_jsonfile

tree_dict = {}  # dictionary that links posts IDs directly to tree object
tree_list = []  # list of convos in tree form (i.e. root is first post)

def map_label_2_int(label):
    label_arr = ['question',
                 'answer',
                 'announcement',
                 'agreement',
                 'appreciation',
                 'disagreement',
                 'negativereaction',
                 'elaboration',
                 'humor',
                 'other']
    return label_arr.index(label)

def map_int_2_label(int_label):
    label_arr = ['question',
                 'answer',
                 'announcement',
                 'agreement',
                 'appreciation',
                 'disagreement',
                 'negativereaction',
                 'elaboration',
                 'humor',
                 'other']
    return label_arr[int_label]

class RedditTree:
    """
    A reddit conversation is a tree. Every post is a tree. 
    Leaf nodes are also trees, but with 0 kids / grandkids
    """
    def __init__(self, body, post_id, label_str):
        
        # Initialize tree variables
        self.body = ''          # body text
        self.post_id = ''       # ID of head
        self.label_str = ''     # Label of root of tree as string
        self.label_int = -1     # Label of root of tree as int
        
        self.children  = []     # number of immediate 
        self.num_child = 0      # immediate children
        self.num_grand = 0      # number of immediate grandkids
        self.tree_size = 1      # total size of tree, including root
        self.depth = 0          # Depth of node. 0 is root.
        self.max_depth = 0      # Depth of entire tree. Min 0 is a single root.
        
        # Set the node's variables based on text and labels first
        self.body = body
        self.post_id = post_id
        self.label_str = label_str
        self.label_int = map_label_2_int(label_str)
        
    
    def adopt_child(self, subTree):
        """
        Attaches a subtree to list of children
        subTree : RedditTree
            a tree that is a child post.
        """
        self.children.append(subTree)
        return
    
    def calc_num_child(self):
        """
        Goes thru tree and counts number of children.
        Request kids to count their children recursively as well
        Calculates and sets count
        """
        self.num_child = len(self.children) # count num of kids
        for child in self.children:         # Go thru all kids
            child.calc_num_child()           # Ask each kid to take stock also
        
    def calc_num_grand(self):
        """
        Goes thru tree and counts number of grandkids
        Request kids to count their grandkids recursively as well
        Calculates and sets count
        """
        self.num_grand = 0      
        for child in self.children:         # for each child
            count = child.get_num_child()   # get number of child's child
            self.num_grand += count         # accumulate the count
            child.calc_num_grand()          # ask each kid to take stock also
    
    def calc_tree_size(self):
        """
        Goes thru tree and counts total num of nodes. Leaf nodes are size 1
        Sets and Returns total number
        """
        if 0==len(self.children):               # if no kids
            self.tree_size = 1                  # tree size is just 1
        else:                                   # if have kids, 
            count = 1
            for child in self.children:         # go thru all kids
                child.calc_tree_size()          # ask kid to count tree size
                count += child.get_tree_size()  # add kid's tree size to count
            self.tree_size = count
    
    def calc_depth(self, depth=0):
        """
        Goes thru tree and assigns the depth to each node. 0 is root
        
        depth : parent's level, optional
            If 0, it means the root post. 
            All other numbers denote the level of post. The default is 0.
        """
        self.depth = depth
        for child in self.children:         # go thru all kids
            new_depth = depth + 1           # increment depth arguement
            child.calc_depth(new_depth)     # ask kid to calculate their depth
    
    def calc_max_depth(self, depth=0):
        """
        Goes thru tree and calculates max depth of the tree. 0 is root.
        Parameters
        ----------
        depth : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        # TODO
        return
    
    def calc_all(self):
        """
        Goes through all functions to calculate all important 
        numbers for the tree
        Calculates num_child, num_grand, tree_size, depth
        """
        self.calc_num_child()
        self.calc_num_grand()
        self.calc_tree_size()
        self.calc_depth()
    
    def get_num_child(self):
        """
        Returns the pre counted number of children quickly
        """
        return self.num_child
    
    def get_num_grand(self):
        """
        Returns the pre counted number of grandchild quickly
        """
        return self.num_grand
    
    def get_tree_size(self):
        """
        Returns the pre counted tree size quickly
        """
        return self.tree_size
    
    def get_depth(self):
        """
        Returns the pre counted depth quickly
        """
        return self.depth
    
    def get_max_depth(self):
        """
        Returns the pre calculated maximum depth of tree quickly
        """
        return self.max_depth
    
def create_link(tree, tree_dict, parent_id):
    try:
        parent_tree = tree_dict[parent_id]
        parent_tree.adopt_child(tree)
    except KeyError:
        pass        

if __name__ == '__main__':
    FILEDIR  = './../data/coarse_discourse/'
    FILENAME = 'coarse_discourse_dump_reddit.json'
    TESTFRACTION = 0.1
    
    entries = extract_jsonfile(FILEDIR, FILENAME)
    test_entries = entries[0:3] # get a subset of trees to deal with
    
    all_posts = []
    # TODO: put into a function
    # TODO: clean blankspace and URLs
    counter = 0                             # counter to track all convos
    for each_thread in test_entries:        # for each convo
        print('Convo %d' % counter)
        posts = each_thread['posts']
        inner_post_counter = 0              # counter to track root post in a thread
        for post in posts:                  # loop through all posts
            try:
                text = post['body']
                name = post['id']
                lbl = post['majority_type']
                
                tree = RedditTree(text,     # create a tree for this post
                                  name, 
                                  lbl)
                
                tree_dict[name] = tree      # add tree into global dictionary
                
                if inner_post_counter == 0: # if root post
                    tree_list.append(tree)  # append tree to global list
                else:                       # else for other posts
                    parent = post['majority_link']
                    create_link(tree,       # find parent tree and create link
                                tree_dict, 
                                parent)
            except Exception:               # encountered key error somewhere
                pass
            inner_post_counter += 1
                
        counter += 1
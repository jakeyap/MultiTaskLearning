#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:27:49 2020

@author: jakeyap
"""
import time
import numpy as np
from scipy import stats 
import torch
from preprocessor_functions import remove_urls, remove_spaces, post_isempty
from convert_coarse_discourse import extract_jsonfile
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from matplotlib.colors import LogNorm

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
        self.tree_size = 1      # total size of self + descendants
        self.depth = 0          # Depth of node. 0 is root.
        self.max_depth = 0      # Depth of entire tree. Min 0 is a single root.
        
        # Set the node's variables based on text and labels first
        self.body = body
        self.post_id = post_id
        self.label_str = label_str
        self.label_int = map_label_2_int(label_str)
    
    def print_text(self):
        """
        Prints out own text, request children to do so as well
        Returns printed stuff
        """
        tabs = '\t' * self.depth
        text = tabs + remove_spaces(self.body[:30])
        print(text)
        for child in self.children:
            text += child.print_text()
        return text
    
    def print_labels(self):
        """
        Prints out labels, request children to do so as well
        Returns printed stuff
        """
        tabs = '\t' * self.depth
        lbls = tabs + self.label_str
        print(lbls)
        for child in self.children:
            lbls += child.print_labels()
        return lbls
    
    def print_depth(self):
        """
        Prints out own depth, request children to do so as well
        Returns printed stuff
        """
        tabs = '\t' * self.depth
        depth = tabs + str(self.depth)
        print(depth)
        for child in self.children:
            depth += child.print_depth()
        return depth
    
    def print_max_depth(self):
        """
        Prints out own depth, request children to do so as well
        Returns printed stuff
        """
        tabs = '\t' * self.depth
        max_depth = tabs + str(self.max_depth)
        print(max_depth)
        for child in self.children:
            max_depth += child.print_max_depth()
        return max_depth
    
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
            child.calc_num_child()          # Ask each kid to take stock also
        
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
        Goes thru tree and counts total num of nodes of self + descendants
        Leaf nodes are size 1
        Calculates and sets total number
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
        Goes thru tree and calculates max depth of the tree from self. 
        0 means leaf nodes.
        """
        max_depth = 0
        if 0==len(self.children):                       # if no kids, it is a leaf
            self.max_depth = 0                          # leaves have maxdepth = 0
        else:
            for child in self.children:                 # if non-leaf
                child.calc_max_depth()
                kid_depth = child.get_max_depth()       # get kids' depth
                max_depth = max(max_depth,kid_depth)    # store the bigger depth
            self.max_depth = max_depth + 1
                
    def calc_all(self):
        """
        Goes through all functions to calculate all important 
        numbers for the tree
        Calculates num_child, num_grand, tree_size, depth
        """
        self.calc_num_child()   # calculate number of children
        self.calc_num_grand()   # calculate number of grandkids
        self.calc_tree_size()   # calculate number of tree size
        self.calc_depth()
        self.calc_max_depth()
    
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
        print('parent not found, skipping')

def split_data(list_of_trees):
    """
    Splits data into 80-10-10 train-dev-test ratio
    Do the split by windows. 
    For every 10 trees, 
        1-8 go into trainset,
        9 goes into devset
        10 goes into testset
    Remainder goes into trainset
    
    Parameters
    ----------
    list_of_trees : a list containing tree objects

    Returns
    -------
    trainset, devset, testset : Each is a list of tree objects
    """
    trainset = []
    devset = []
    testset = []
    
    count = 0
    datalen = len(list_of_trees)
    while count < datalen:
        if count % 10 == 9:
            testset.append(list_of_trees[count])
        elif count % 10 == 8:
            devset.append(list_of_trees[count])
        else:
            trainset.append(list_of_trees[count])
        count += 1
    return trainset, devset, testset

def save_trees(list_of_trees, filename):
    """
    Saves a list of trees into a file

    Parameters
    ----------
    list_of_trees : list of trees
    filename : string of filename to save list to
    """
    torch.save(list_of_trees, filename)

def reload_trees(filename):
    """
    Reloads a list of trees from a file
    
    Parameters
    ----------
    filename : string of filename to load from

    Returns a list of trees
    """
    return torch.load(filename)

def build_trees(json_entries):
    """
    Builds the trees from the coarse discourse dataset

    Parameters
    ----------
    json_entries : extracted lines from the json file as dict

    Raises
    ------
    KeyError : if cannot find body 

    Returns
    -------
    main_trees, all_trees_dict.
    main_trees : list of tree objects. each tree is a conversation in reddit
    all_tree_dict : dictionary of every post. key = postid, value = tree object

    """
    
    all_trees_dict = {}                     # dict of every post. K=posts IDs. V=tree objects
    main_trees = []                         # list of every convo in tree form
    counter = 0                             # counter to track all convos
    for each_thread in json_entries:        # for each convo
        print('Convo %d' % counter)
        local_trees_dict = {}               # dict of every post in this convo. K=post IDs, V=tree objects. For quick reference
        posts = each_thread['posts']
        inner_post_counter = 0              # counter to track root post in a thread
        for post in posts:                  # loop through all posts to create trees
            try:
                if inner_post_counter == 0: # if root post, stick title with body 
                    text = each_thread['title'] + post['body']
                else:                       # if not root post, just get body
                    text = post['body']     
                name = post['id']           # get the post ID
                lbl = post['majority_type'] # get the post label
                
                text = remove_spaces(text)  # remove spaces
                text = remove_urls(text)    # remove URLs
                if post_isempty(text):
                    print('empty post')     # skip empty posts
                    raise KeyError
                if '[deleted]'==text:
                    print('deleted post')   # skip deleted posts
                    raise KeyError
                tree = RedditTree(text,     # create a tree for this post
                                  name, 
                                  lbl)
                
                all_trees_dict[name] = tree # add tree into global, local dictionaries
                local_trees_dict[name] = tree
                if inner_post_counter == 0: # if root post
                    main_trees.append(tree) # append tree to global list
                else:                       # else for other posts
                    parent = post['majority_link']
                    create_link(tree,       # find parent tree and create link
                                local_trees_dict, 
                                parent)
            except Exception:               # encountered key error somewhere
                pass
            inner_post_counter += 1
                
        counter += 1
        for eachtree in main_trees:
            eachtree.calc_all()    
    return main_trees, all_trees_dict

def extract_trees_by_level(list_of_trees):
    '''
    Spits out a list of lists of trees

    Parameters
    ----------
    list_of_trees : list of root level trees

    Returns
    -------
    all_lvl_trees : list of list of trees
        [[root level trees]
         [lvl 1 trees]
         [lvl 2 trees]...]

    '''
    # list of lists. each list contains entire trees of that level
    # for example, all_lvl_trees[0] are all trees starting at root
    # all_lvl_trees[1] are all trees starting at level 1
    max_depth = -1
    for tree in list_of_trees:
        max_depth = max(max_depth, tree.max_depth)
    
    print('======== Creating empty lists ========')
    all_lvl_trees = [list_of_trees,]
    for i in range(max_depth):
        all_lvl_trees.append([])
    print('======== Collecting trees by level ========')
    for i in range (1, max_depth):
        print("Collecting level %d's trees" % i)
        prev_lvl_trees = all_lvl_trees[i-1]
        this_lvl_trees = all_lvl_trees[i]
        for tree in prev_lvl_trees:
            this_lvl_trees.extend(tree.children)
    return all_lvl_trees
    
    
def profile_dataset_by_lvl(list_of_trees, maxdepth=5):
    ''' 
    plots average stats per level, up to maxdepth. mean,median,mode of the following
    tree_size, num_kids, num_grands, fam_size
    '''
    FONTSIZE = 13
    TITLESIZE = 15
    # list of lists. each list contains entire trees of that level
    # for example, all_lvl_trees[0] are all trees starting at root
    # all_lvl_trees[1] are all trees starting at level 1
    
    all_lvl_trees = [list_of_trees]
    print('======== Creating empty lists ========')
    for i in range (maxdepth-1):
        all_lvl_trees.append([])
    
    print('======== Collecting trees by level ========')
    for i in range (1, maxdepth):
        print("Collecting level %d's trees" % i)
        prev_lvl_trees = all_lvl_trees[i-1]
        this_lvl_trees = all_lvl_trees[i]
        for tree in prev_lvl_trees:
            this_lvl_trees.extend(tree.children)
        
    # Each element is an array for each level's data
    data_list = []
    print('======== Recording statistics ========')
    for i in range(maxdepth):
        print('Recording level %d trees' % i)
        this_lvl_trees = all_lvl_trees[i]
        # Arrays for counting variables. Rows=trees
        # Cols= 0:tree_size, 1:num_kids, 2:num_grands, 3:fam_size
        raw_stats_arr = np.zeros((len(this_lvl_trees), 4))
        
        # go thru each tree at this level, count the stats
        num_trees = len(this_lvl_trees)
        for j in range(num_trees):
            tree = this_lvl_trees[j]
            num_child = tree.num_child
            num_grand = tree.num_grand
            tree_size = tree.tree_size
            fam_size = num_child + num_grand
            raw_stats_arr[j,0] = tree_size
            raw_stats_arr[j,1] = num_child
            raw_stats_arr[j,2] = num_grand
            raw_stats_arr[j,3] = fam_size
        data_list.append(raw_stats_arr)
    
    tree_sizes = np.zeros((maxdepth,3))
    num_childs = np.zeros((maxdepth,3))
    num_grands = np.zeros((maxdepth,3))
    fam_sizes = np.zeros((maxdepth,3)) 
    
    for i in range(maxdepth):
        raw_stats_arr = data_list[i] # (n,4) array where n=num of trees
        mean = np.average(raw_stats_arr, 0) # shape=(4,)
        medi = np.median(raw_stats_arr, 0)  # shape=(4,)
        mode = stats.mode(raw_stats_arr, 0).mode # shape=(4,)
        
        mean = mean.reshape((4,-1))
        medi = medi.reshape((4,-1))
        mode = mode.reshape((4,-1))
        
        combined = np.concatenate((mean, medi, mode), axis=1) # shape=(4,3)
        tree_sizes[i, 0:3] = combined[0, 0:3]
        num_childs[i, 0:3] = combined[1, 0:3]
        num_grands[i, 0:3] = combined[2, 0:3]
        fam_sizes[i, 0:3] = combined[3, 0:3]
    
    horz = np.arange(0, maxdepth, 1)
    fig, axes = plt.subplots(4, 1, sharex=True)
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    ax3 = axes[3]
    
    ax0.set_title('Stats by level', size=TITLESIZE)
    ax0.set_ylabel('Sub-tree sizes', size=FONTSIZE)
    ax0.plot(horz, tree_sizes[:,0], marker='o', label='mean')
    ax0.plot(horz, tree_sizes[:,1], marker='+', label='median')
    ax0.plot(horz, tree_sizes[:,2], marker='x', label='mode')
    ax0.legend()
    ax0.grid(True)
    ax1.set_ylabel('Num kids', size=FONTSIZE)
    ax1.plot(horz, num_childs[:,0], marker='o', label='mean')
    ax1.plot(horz, num_childs[:,1], marker='+', label='median')
    ax1.plot(horz, num_childs[:,2], marker='x', label='mode')
    ax1.legend()
    ax1.grid(True)
    ax2.set_ylabel('Num grandkids', size=FONTSIZE)
    ax2.plot(horz, num_grands[:,0], marker='o', label='mean')
    ax2.plot(horz, num_grands[:,1], marker='+', label='median')
    ax2.plot(horz, num_grands[:,2], marker='x', label='mode')
    ax2.legend()
    ax2.grid(True)
    ax3.set_ylabel('Num kids + grandkids', size=FONTSIZE)
    ax3.plot(horz, fam_sizes[:,0], marker='o', label='mean')
    ax3.plot(horz, fam_sizes[:,1], marker='+', label='median')
    ax3.plot(horz, fam_sizes[:,2], marker='x', label='mode')
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlabel('Distance from root', size=TITLESIZE)
    
    stats_dict = {'tree_sizes' : tree_sizes,
                  'num_childs' : num_childs,
                  'num_grands' : num_grands,
                  'fam_sizes' : fam_sizes}
    return all_lvl_trees, stats_dict

def profile_dataset(list_of_trees, title='root'):
    ''' goes thru dataset to count the number of kids/grandkids for posts '''
    ''' only run numbers for root posts '''
    TITLESIZE = 15
    FONTSIZE = 13
    tree_sizes = [] # for tracking total tree sizes
    max_depths = [] # for tracking max depths of convos
    num_kids = []   # for tracking num of 1st level nodes
    num_grands = [] # for tracking num of 2nd level nodes
    fam_sizes = []  # for tracking num of 1st + 2nd level nodes
    
    for tree in list_of_trees:
        tree_size = tree.get_tree_size()    # get the max size of tree
        max_depth = tree.get_max_depth()    # get max depth of tree
        kids = tree.get_num_child()         # get num of children
        grands = tree.get_num_grand()       # get num of grandkids
        family_size = kids + grands         # size of children + grandkids
        
        tree_sizes.append(tree_size)
        max_depths.append(max_depth)
        num_kids.append(kids)
        num_grands.append(grands)
        fam_sizes.append(family_size)
    
    fig0, axes0 = plt.subplots(4,2, sharex=True)
    ax0 = axes0[0][1]
    ax1 = axes0[1][1]
    ax2 = axes0[2][1]
    ax3 = axes0[3][1]
    ax4 = axes0[0][0]
    ax5 = axes0[1][0]
    ax6 = axes0[2][0]
    ax7 = axes0[3][0]
    
    # for controlling plot ranges
    depth_range = [[-1,45],[-1,14]]
    child_range = [[-1,45],[-1,41]]
    grand_range = [[-1,45],[-1,19]]
    total_range = [[-1,45],[-1,45]]
    # for controlling histogram binning
    depth_bins = [23,15]
    child_bins = [23,21]
    grand_bins = [23,10]
    total_bins = [23,23]
    # copy the default color map
    my_cmap = copy.copy(colormap.get_cmap('viridis'))
    # set the NaNs to lowest color
    my_cmap.set_bad(my_cmap.colors[0])
    ax4.set_title('Possible combos at ' + title, size=TITLESIZE)
    ax4.scatter(tree_sizes, max_depths)
    ax4.grid(True)
    ax4.set_ylim(depth_range[1])
    ax4.set_xlim(depth_range[0])
    ax4.set_ylabel('Depth', size=FONTSIZE)
    
    ax5.scatter(tree_sizes, num_kids)
    ax5.grid(True)
    ax5.set_ylim(child_range[1])
    ax5.set_xlim(child_range[0])
    ax5.set_ylabel('Num kids', size=FONTSIZE)
    
    ax6.scatter(tree_sizes, num_grands)
    ax6.grid(True)
    ax6.set_ylim(grand_range[1])
    ax6.set_xlim(grand_range[0])
    ax6.set_ylabel('Num grandkids', size=FONTSIZE)
    
    ax7.scatter(tree_sizes, fam_sizes)
    ax7.grid(True)
    ax7.set_ylim(total_range[1])
    ax7.set_xlim(total_range[0])
    ax7.set_ylabel('Num kids + grandkids', size=FONTSIZE)
    
    ax7.set_xlabel('Convo Tree Size', size=TITLESIZE)
    
    ax0.set_title('2D h.gram of combos at ' + title, size=TITLESIZE)
    data0, xbins0, ybins0, handle0 = ax0.hist2d(tree_sizes, max_depths,
                                                norm=LogNorm(vmin=1),
                                                cmap=my_cmap,
                                                bins=depth_bins,
                                                range=depth_range)
    plt.colorbar(handle0, ax=ax0, shrink=0.6, pad=0.05)
    ax0.set_ylabel('Depth', size=FONTSIZE)
    ax0.grid(True)
    
    ax1.set_ylim([-1,40])
    data1, xbins1, ybins1, handle1 = ax1.hist2d(tree_sizes, num_kids,
                                                norm=LogNorm(vmin=1),
                                                cmap=my_cmap,
                                                bins=child_bins,
                                                range=child_range)
    plt.colorbar(handle1, ax=ax1, shrink=0.6, pad=0.05)
    ax1.set_ylabel('Num kids', size=FONTSIZE)
    ax1.grid(True)
    
    data2, xbins2, ybins2, handle2 = ax2.hist2d(tree_sizes, num_grands,
                                                norm=LogNorm(vmin=1),
                                                cmap=my_cmap,
                                                bins=grand_bins,
                                                range=grand_range)
    plt.colorbar(handle2, ax=ax2, shrink=0.6, pad=0.05)
    ax2.set_ylabel('Num grandkids', size=FONTSIZE)
    ax2.grid(True)
    
    ax3.set_ylim([-1,40])
    data3, xbins3, ybins3, handle3 = ax3.hist2d(tree_sizes, fam_sizes,
                                                norm=LogNorm(vmin=1),
                                                cmap=my_cmap,
                                                bins=total_bins,
                                                range=total_range)
    plt.colorbar(handle3, ax=ax3, shrink=0.6, pad=0.05)
    ax3.set_ylabel('Num kids + grandkids', size=FONTSIZE)
    ax3.grid(True)
    ax3.set_xlabel('Convo Tree Size', size=TITLESIZE)
    
    
    maximize_figs()
    
    return tree_sizes, max_depths, num_kids, num_grands, fam_sizes

def plot_family_vs_tree_sizes(list_of_trees, title=''):
    ''' Generates plot of kids + grandkids count vs entire tree size'''
    fam_sizes = []
    tree_sizes = []
    for tree in list_of_trees:
        fam_sizes.append(tree.num_child + tree.num_grand)
        tree_sizes.append(tree.tree_size)
    
    # copy the default color map
    my_cmap = copy.copy(colormap.get_cmap('viridis'))
    # set the NaNs to lowest color
    my_cmap.set_bad(my_cmap.colors[0])
    
    fig, axes = plt.subplots(1,2)
    ax0 = axes[0]
    ax1 = axes[1]
    ax0.scatter(fam_sizes, tree_sizes)
    ax0.grid(True)
    data, xbins, ybins, handle = ax1.hist2d(fam_sizes, tree_sizes, norm=LogNorm(), cmap=my_cmap)
    ax1.grid(True)
    plt.colorbar(handle, ax=ax1)
    plt.suptitle('Tree size vs Immediate family size at '+title, size=15)
    ax0.set_xlabel('Num kids + grandkids', size=13)
    ax1.set_xlabel('Num kids + grandkids', size=13)
    ax0.set_ylabel('Tree size', size=13)
    

def maximize_figs():
    figures = plt.get_fignums()
    for each_figure in figures:
        plt.figure(each_figure)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()

def tighten_plots():
    figures = plt.get_fignums()
    for each_figure in figures:
        plt.figure(each_figure)
        plt.tight_layout()

def print_tree(list_of_trees, number=-1, thing_to_print='depth'):
    if number == -1:
        number = np.random.randint(0, len(list_of_trees))
        tree = list_of_trees[number]
    else:
        tree = list_of_trees[number]
    if thing_to_print=='depth':
        print('Printing depth of nodes')
        tree.print_depth()
    if thing_to_print=='max_depth':
        print('Printing max of subtrees')
        tree.print_max_depth()
    if thing_to_print=='labels':
        print('Printing labels')
        tree.print_labels()
    if thing_to_print=='text':
        print('Printing text in tree')
        tree.print_text()

def pack_into_tsv_files(breadth=3, depth=1):
    '''
    Packs trees into TSV files in the same format as before, where posts are
    separated by ' ||||| ' in the text file.. 
    Elements in each tsv line follow this format
    
    root ||||| post-1 ||||| 
    
    Returns
    -------
    None.

    '''
    # TODO: build into TSV files
    
    return

def encode_and_tokenize_tsv():
    '''
    Encodes and tokenizes tsv files

    Returns
    -------
    None.

    '''
    # TODO: implement this
    return

if __name__ == '__main__':
    
    FILEDIR  = './../data/coarse_discourse/'
    FILENAME = 'coarse_discourse_dump_reddit.json'
    trainset = reload_trees(FILEDIR + 'full_trees/full_trees_train.pkl')
    devset = reload_trees(FILEDIR + 'full_trees/full_trees_dev.pkl')
    testset = reload_trees(FILEDIR + 'full_trees/full_trees_test.pkl')
    
    fullset = []
    fullset.extend(trainset)
    fullset.extend(devset)
    fullset.extend(testset)
    
    all_lvl_trees = extract_trees_by_level(fullset)
    ''' ======= 2020 Dec 28, code here to convert text to trees, store in pkl files ======= '''
    '''
    entries = extract_jsonfile(FILEDIR, FILENAME)
    #entries = entries [0:300] # try a subset first
    time1 = time.time()
    print('======= Building trees =======')
    main_trees, all_trees = build_trees(entries)
    print('======= Splitting data =======')
    train_set, dev_set, test_set = split_data(main_trees)
    print('======== Saving trees ========')
    save_trees(train_set, FILEDIR+'full_trees_train.pkl')
    save_trees(dev_set, FILEDIR+'full_trees_dev.pkl')
    save_trees(test_set, FILEDIR+'full_trees_test.pkl')
    time2 = time.time()
    minutes = (time2-time1) // 60
    seconds = (time2-time1) % 60
    print('%d minutes %2d seconds' % (minutes, seconds))
    '''
    
    ''' ======= 2020 Dec 28, code here to plot mean, median, mode ======= ''' 
    ''' all_lvl_trees_v2, stats_dict = profile_dataset_by_lvl(fullset,5) '''
    
    ''' ======= 2020 Dec 28, code here to plot tree stats per level ======= '''
    '''
    root_trees = all_lvl_trees[0]
    lvl1_trees = all_lvl_trees[1]
    lvl2_trees = all_lvl_trees[2]
    lvl3_trees = all_lvl_trees[3]
    lvl4_trees = all_lvl_trees[4]
    
    lvl0_data = profile_dataset(root_trees, 'root')
    lvl1_data = profile_dataset(lvl1_trees, 'lvl1')
    lvl2_data = profile_dataset(lvl2_trees, 'lvl2')
    lvl3_data = profile_dataset(lvl3_trees, 'lvl3')
    lvl4_data = profile_dataset(lvl4_trees, 'lvl4')
    maximize_figs()
    tighten_plots()
    '''
    ''' ======= 2020 Dec 29, code here to count posts for different truncation strategy ======= '''
    '''
    counts = []
    for each_list in all_lvl_trees:
        count = 0
        for tree in each_list:
            count += 1
        counts.append(count)
    # count what i have left after pruning [deleted] or missing posts
    print('counts: ' + str(counts))
    total_posts = sum(counts)
    print('sum:  %d ' % total_posts)
    
    root_trees = all_lvl_trees[0]
    sizes = []
    filt_root = []
    filt_lvl1a = []
    filt_lvl1b = []
    filt_lvl1c = []
    filt_lvl2a = []
    filt_lvl2b = []
    filt_lvl2c = []
    for tree in root_trees:
        filt_root.append(tree)
        sizes.append(tree.tree_size)
        for child in tree.children[:4]:
            filt_lvl1a.append(child)
            for grand in child.children[:1]:
                filt_lvl2a.append(grand)
        
        if len(tree.children)>=4:
            sizes.append(tree.tree_size)
            for child in tree.children[4:8]:
                filt_lvl1b.append(child)
                for grand in child.children[:1]:
                    filt_lvl2b.append(grand)
        
        if len(tree.children)>=8:
            sizes.append(tree.tree_size)
            for child in tree.children[8:12]:
                filt_lvl1c.append(child)
                for grand in child.children[:1]:
                    filt_lvl2c.append(grand)
                
    print('head:  \t \t \t%5d' % len(filt_root))
    print('lvl1: %5d \t%5d \t%5d' % (len(filt_lvl1a), len(filt_lvl1b), len(filt_lvl1c)))
    print('lvl2: %5d \t%5d \t%5d' % (len(filt_lvl2a), len(filt_lvl2b), len(filt_lvl2c)))
    
    print('sizes median %2d mean %.2f std %.2f' % (np.median(sizes), np.average(sizes), np.std(sizes)))
    
    plot_family_vs_tree_sizes(all_lvl_trees[0], 'root')
    plot_family_vs_tree_sizes(all_lvl_trees[1], 'lvl1')
    plot_family_vs_tree_sizes(all_lvl_trees[2], 'lvl2')
    maximize_figs()
    '''
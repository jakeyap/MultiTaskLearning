The 3 full_trees_XXX.pkl files here contain the conversations in tree object forms.
They are not tokenized/encoded yet

encoded_dict.pkl stores encoded data as a big dictionary. The max length of each sentence is 512. Shorter sentences are padded
    Each key is a post ID. 
    Each value is another dictionary, where the keys are 
    'input_ids'
    'token_type_ids'
    'attention_mask'

The median tree size in this dataset is 7. 
It is smaller than previous value of 8 because the [deleted] posts are pruned.

For the whole dataset, after [deleted] and empty are pruned, there are a total of 89477 posts. 
Breakdown is as follows
root    9201
lvl1    44421
lvl2    18645
lvl3    9202
lvl4    4143
lvl5    1953
lvl6    945
lvl7    486
lvl8    260
lvl9    132
lvl10   85
lvl11   4

========================================================================================================================
====================================================  Approach 3  ======================================================
====================== For each of the examples in this section, only take 1 grandchild per child ======================
Strategy 1 in code
After pruning (width=3, depth=2) stride size=3
1 strides=3, 42708 (47.7%) of data is retained.
2 strides=3, 56260 (62.8%) of data is retained. 
3 strides=3, 62092 (69.3%) of data is retained.  
Breakdown is as follows
root    9201
lvl1a   23053   lvl1b   10423   lvl1c   4751
lvl2a   10454   lvl2b   3129    lvl2c   1081

Each example is formatted as 
[root] [child1] [grandkid1] [child2] [grandkid2] [child3] [grandkid3]

Strategy 2 in code
After pruning (width=4, depth=2), stride size=4
1 strides=4, 48654 (54.3%) of data is retained.
2 strides=4, 60596 (67.7%) of data is retained. 
3 strides=4, 65069 (71.6%) of data is retained.  
Breakdown is as follows
root    9201
lvl1a   27570   lvl1b   9420    lvl1c   3742
lvl2a   11883   lvl2b   2522    lvl2c   731

Each example is formatted as 
[root] [child1] [grandkid1] [child2] [grandkid2] [child3] [grandkid3] [child4] [grandkid4]

Fam sizes median  5 mean 6.85 std 5.98
Tree sizes median  7 mean 9.72 std 7.61
========================================================================================================================
====================================================  Approach 4  ======================================================
For each tree, randomly pick 3 child per cycle. Each child brings 1 grandchild along.
Fam 




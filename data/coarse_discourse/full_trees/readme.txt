The 3 full_trees_XXX.pkl files here contain the conversations in tree object forms.
They are not tokenized/encoded yet

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

After pruning (width=3, depth=2) there are 42708 posts left. 47.7% of data is retained. Breakdown is as follows
root    9201
lvl1    23053
lvl2    10454

After pruning (width=4, depth=2) there are 42708 posts left. 54.3% of data is retained. Breakdown is as follows
root    9201
lvl1    27570
lvl2    11883

After pruning (width=4, depth=2, 1 horz stride=4 @ lvl1), 67.7% of data is retained. Breakdown is as follows
root    9201
lvl1a   27570   lvl1b   9420
lvl2a   11883   lvl2b   2522

After pruning (width=4, depth=2, 2 horz strides=4 @ lvl1), 71.6% of data is retained. Breakdown is as follows
root    9201
lvl1a   27570   lvl1b   9420    lvl1c   3742
lvl2a   11883   lvl2b   2522    lvl2c   731

========================================================================================================================
====================================================  Approach 4  ======================================================


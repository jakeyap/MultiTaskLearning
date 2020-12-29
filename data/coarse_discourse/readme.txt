coarse_discourse_dump_reddit.json   This is the dataset's released raw datafile
coarse_discourse_dump_reddit.tsv    TSV file of the root + level 1 nodes. [deleted] are kept. URLs are masked using [URL] token. 

/full_trees                         Contains files that stores convos as tree objects. Trees with nodes starting with [deleted] are pruned. URLs are masked using [URL] token
                                    To use .pkl files inside, use functions in /utilities/handle_coarse_discourse_tree.py

/flattened                          Contains files that stores convos as root-2-leaf paths. [deleted] posts are retained



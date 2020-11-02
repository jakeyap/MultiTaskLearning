# ***MultiTaskLearning Repo***

This repo is to experiment with multitask learning. There are 2 tasks at hand.    
The performance of the joint training should be better than doing tasks separately.    

### **Tasks**    
**1. Length Prediction of Conversation Threads:**    
Given some starting posts in a thread, predict how long the thread will last.    
This will be framed as a binary classfication problem.

**2. Stance Analysis:**    
Figure out the stance of social media posts.    
This is a multiclass classification problem.

### **Models**     
The base model used is shown here.

<img src="./misc/ModelA0.png" alt="drawing" width="600"/>    

The BERT attention layer tried so far are combos of 1-4 layers.    
    
### **Datasets used**        
There were 2 datasets used in this work
    
**1. SemEval17 (Specifically, Task 8-Subtask A)**    
Tweets are labelled according to their stance towards a parent post    
    
    0 = Deny
    1 = Support
    2 = Query
    3 = Comment

Here's how their label density looks like

<img src="./data/semeval17/label_density.png" alt="drawing" width="600"/>    

**2. Coarse Discourse Dataset (From Convokit)**    
A bunch of reddit threads.
Posts have the following labels

    Question
    Answer
    Announcement
    Agreement
    Appreciation
    Disagreement
    Negative Reaction
    Elaboration
    Humor
    Other
    
Here's how their label density looks like

<img src="./data/coarse_discourse/category_density.png" alt="drawing" width="600"/>    

### **Pre-processing of datasets**    
The first thing to do is to merge both datasets into 1.    
Convert Coarse Discourse labels into SemEval labels as follows.    

    Question      --> Query
    Answer        --> Comment
    Announcement  --> Comment
    Agreement     --> Support
    Appreciation  --> Comment
    Elaboration   --> Comment
    Humor         --> Comment
    Other         --> Comment
    Disagreement  --> Deny
    Neg. Reaction --> Deny

Next, take care of thread lengths. Here's a histogram of the distribution in the training set.    
The x-axis represents thread length., y-axis means length frequency.    

<img src="./data/combined/training_convo_lengths_pre_post_prune.png" alt="drawing" width="600"/>    

To define whether a conversation is long, I split it along the median or 9.     
i.e. Threads with lengths. >=9 are long
Backstrom et. al. (2013) split their facebook data at 8, also a median length.    

I decided to look at the first N number of posts, then predict whether the conversation will be "long". I set N=4.    
Zhang et. al. (2018) set N=2 only for their toxicity prediction task.

Each thread is now framed as    
- Post0
- Reply1
- Reply2
- Reply3
- Final thread length
- Post0 stance label
- Reply1 stance label
- Reply2 stance label
- Reply3 stance label

Since N is 4, any thread that is length 2 or 3 is problematic when it comes to training for stance classification.    
To handle such situations, I stuck in empty posts and created a annotated them with a new label - 'isEmpty'

10% used as a test set    
10% used as a validation set (fixed, k-folds not implemented)    
80% used as training set

### **Other Important Stuff**    

Here's a summary of the stuff tried. It is just a short explnation. See ./misc/experiment_results.ods for full settings of each experiment and full results.

*Optimizers:*   
Tried *Base SGD*, *ADAM*.    

*Loss Functions:*   
Tried uniform cross entropy loss (multiclass for stance, binary for length)    
as well as weighted cross entropy for stance (1, 10, 1, 5, 1). Deny=10, query=5    

*Learning rates (LR)*    
Most of the experiments have LR set to 
\\[10^{-4}, 2 \times 10^{-4}, 5 \times 10^{-4}\\]
Some have dual learning rates. See excel file for actual data.

### **Experiment Results**    

Training using base SGD optimizers didn't work for stance prediction. 
Cannot learn DENY class. Probably too slow or something. 
Training with ADAM seem to work better

Full set of results stored in ./misc/experiment_results.ods.
Results shown below are the more interesting ones. 

**Experiment 12 (Best in Deny F1 score)**   
ModelB4
* 4 transformer layers stacked on top of BERT
* Weighted loss function for stance
* Looked up 4 posts/thread, 256 tokens/post
* Minibatch size=15
* num GPUs=5
* Training took 5h15m

Labels |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.1832|0.3077|**0.2297**|78   
Support|0.8888|0.8130|0.8492|1032
Query  |0.6588|0.8058|0.7249|242
Comment|0.8999|0.8922|0.8961|2227
F1 avg |      |      |0.7400|
F1 wo isEmpty||      |0.6750|
Acc.   |      |      |86.4%|

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.5887|0.5321|0.5589|468
Long   |0.6075|0.6608|0.6331|513
Average|      |      |0.5960
Accuracy|     |      |59.9%


**Experiment 13 (Best in Overall Stance F1 score)**   
ModelB3
* *3* transformer layers stacked on top of BERT
* Weighted loss function for stance
* Looked up 4 posts/thread, 256 tokens/post
* Minibatch size=15
* num GPUs=5
* Training took 5h

Labels |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.1770|0.2564|0.2094|78   
Support|0.8843|0.8295|0.8560|1032
Query  |0.6656|0.8884|0.7611|242
Comment|0.9099|0.8886|0.8991|2227
F1 avg |      |      |**0.7451**|
F1 wo isEmpty||      |**0.6814**|
Acc.   |      |      |87.0%|

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.7073|0.5577|0.6237|468
Long   |0.6618|0.7895|0.7200|513
Average|      |      |0.6718
Accuracy|     |      |67.9%

**Experiment 20 (Best in Length acc., OK in stance)**   
ModelC4
* *4* transformer layers stacked on top of BERT
* Weighted loss function for stance
* *3x higher starting LR for length vs stance tasks*
* Looked up 4 posts/thread, 256 tokens/post
* Minibatch size=6 
* num GPUs=2
* Training took 6h

Labels |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.1400|0.3590|0.2014|78   
Support|0.8539|0.7190|0.7806|1032
Query  |0.4444|0.9256|0.6005|242
Comment|0.9123|0.8217|0.8646|2227
F1 avg |      |      |0.6895|
F1 wo isEmpty||      |0.6118|
Acc.   |      |      |80.8%|

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.8097|0.4637|0.5897|468
Long   |0.680|0.9006|0.7537|513
Average|      |      |0.6717
Accuracy|     |      |**69.2%**



### **Discussion**

So far, I am near / exceeding Backstrom's length prediction accuracy. But the stance prediction task isn't very good.    
There are more modern papers, but I don't really understand their metrics. An example is the Spearman Rho score in Kowalczyk et. al. (2019)    

In Kowalczyk et. al. (2019), they framed the problem as a regression problem. So their metrics of error are different I guess.    
Accuracy isn't relevant in their use case. But they did use other features to get their results (eg. follower count, num likes, account age etc.)    

In the various ModelCs, I tried to make the length task learn faster compared to the stance task. I did it by forcing the learning rate of the length loss to be 3x that of the stance loss. I'm not sure whether ADAM will screw up this hardcoded behavior though.    

Perhaps it might be easier to do double stepping the length prediction task. In pseudo code:

1. *for each minibatch:*    
    1.1 *learn the length*    
    1.2 *learn the stance*    
    1.3 *learn the length again*

### **Further steps**    
1. Strengthen the model
    * ~~Stack attention layers higher~~
    * ~~Restore URLs~~ [Doesn't seem necessary. Some papers keep URLs, others filter them]
    * Double stepping the learning for length
2. Abalation Study
    * Train the network for stance only
    * Train the network for length prediction only
    * Remove middle attention network, straight into MLPs
3. Build another entirely different model
    * 2 entirely separate networks to not discard posts. Share only underlying BERT

## **References**
[1] Backstrom, L., Kleinberg, J., Lee, L., & Danescu-Niculescu-Mizil, C. (2013). Characterizing and curating conversation threads: Expansion, focus, volume, re-entry. WSDM 2013 - Proceedings of the 6th ACM International Conference on Web Search and Data Mining, 13–22. https://doi.org/10.1145/2433396.2433401

[2] Zhang, J., Chang, J. P., Danescu-Niculescu-Mizil, C., Dixon, L., Thain, N., Hua, Y., & Taraborelli, D. (2018). Conversations gone awry: Detecting early signs of conversational failure. ACL 2018 - 56th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference (Long Papers), 1, 1350–1361. https://doi.org/10.18653/v1/p18-1125

[3] Kowalczyk, D. K., & Larsen, J. (2019). Scalable privacy-compliant virality prediction on twitter? CEUR Workshop Proceedings, 2328(Cohen), 12–27.
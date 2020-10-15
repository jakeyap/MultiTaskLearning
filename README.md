# **MultiTaskLearning Repo**

This repo is to experiment with multitask learning. There are 2 tasks at hand.    

After reading about multitask learning, my hypothesis is that the performance of the joint training should be better than doing tasks separately.

### **Tasks**
**1. Length Prediction of Conversation Threads:**    
Given some starting posts in a thread, predict how long the thread will last.    
This will be framed as a binary classfication problem.

**2. Stance Analysis:**    
Figure out the stance of social media posts.    
This is a multiclass classification problem.

    
### **Datasets used**
    
There were 2 datasets used in this work
    
**1. SemEval17 (Specifically, Task 8-Subtask A)**    

Tweets are labelled according to their stance towards a parent post

    0 = Deny
    1 = Support
    2 = Query
    3 = Comment

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
    
### **Pre-processing of datasets**

The first thing to do is to merge both datasets into 1 

### **Datasets used**
Models used

    Steps
    1. Do both tasks jointly
    2. Do both tasks individually

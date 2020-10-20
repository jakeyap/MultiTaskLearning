#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:38:09 2020

@author: jakeyap
"""
# ***Old Results***

**Experiment 1**     

Stance |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.0000|0.0000|0.0000|78   
Support|0.8063|0.7984|0.8023|1032
Query  |0.7647|0.7521|0.7583|242
Comment|0.8851|0.9133|0.8990|2227
Average|      |      |0.6914

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.7575|0.4872|0.5930|468
Long   |0.6471|0.8577|0.7376|513   
Average|      |      |0.6653
Accuracy|     |      |68.1%


**Experiment 2**    

Labels |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.0000|0.0000|0.0000|78   
Support|0.8411|0.7849|0.8120|1032
Query  |0.6870|0.7438|0.7143|242
Comment|0.8776|0.9273|0.9017|2227
Average|      |      |0.6661

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.6826|0.5791|0.6266|468
Long   |0.6627|0.7544|0.7056|513   
Average|      |      |0.6856
Accuracy|     |      |67.1%

**Experiment 3**

Labels |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
isEmpty|1.0000|1.0000|1.0000|345
Deny   |0.5000|0.0256|0.0488|78   
Support|0.8618|0.8275|0.8443|1032
Query  |0.6955|0.7645|0.7283|242
Comment|0.8909|0.9273|0.9087|2227
Average|      |      |0.7060

Length |Precision|Recall|F1 score|Support
-------|---------|------|--------|-------
Short  |0.7143|0.5235|0.6042|468
Long   |0.6505|0.8090|0.7211|513
Average|      |      |0.6627
Accuracy|     |      |67.3%

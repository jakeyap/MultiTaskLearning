#!/usr/bin/env bash
#@author: jakeyap on 20201013 1130pm

conda activate env1
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --N_EPOCHS=20 --BATCH_SIZE_TRAIN=24 --LOG_INTERVAL=10 --LEARNING_RATE=0.0005 --MOMENTUM=0.125

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --N_EPOCHS=20 --BATCH_SIZE_TRAIN=24 --LOG_INTERVAL=10 --LEARNING_RATE=0.0005 --MOMENTUM=0.250

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python main.py --N_EPOCHS=20 --BATCH_SIZE_TRAIN=24 --LOG_INTERVAL=10 --LEARNING_RATE=0.0005 --MOMENTUM=0.500

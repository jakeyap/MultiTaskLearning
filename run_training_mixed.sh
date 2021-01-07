#!/usr/bin/env bash
#@author: jakeyap on 20201013 1130pm

    
# exp04
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=4 \
    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=512 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM' --DO_TRAIN  \
    --MODELNAME='alt_ModelF4' --NAME='exp04' \
    --WEIGHTED_STANCE --DOUBLESTEP
    
# exp03
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=4 \
    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=512 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM' --DO_TRAIN  \
    --MODELNAME='alt_ModelF3' --NAME='exp03' \
    --WEIGHTED_STANCE --DOUBLESTEP
        
# exp02
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=4 \
    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=512 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM' --DO_TRAIN  \
    --MODELNAME='alt_ModelF2' --NAME='exp02' \
    --WEIGHTED_STANCE --DOUBLESTEP

# exp01
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=4 \
    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=512 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM' --DO_TRAIN  \
    --MODELNAME='alt_ModelF1' --NAME='exp01' \
    --WEIGHTED_STANCE --DOUBLESTEP
    
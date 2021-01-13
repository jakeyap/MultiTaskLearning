#!/usr/bin/env bash
#@author: jakeyap on 20201013 1130pm

# exp12
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=20 --BATCH_SIZE_TEST=20 --LOG_INTERVAL=10 --N_EPOCHS=8 \
    --LEARNING_RATE=0.00005 --MAX_POST_LENGTH=128 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM_V' --DO_TRAIN  \
    --MODELNAME='alt_ModelF4' --NAME='exp12' \
    --WEIGHTED_STANCE

# exp11
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=20 --BATCH_SIZE_TEST=20 --LOG_INTERVAL=10 --N_EPOCHS=8 \
    --LEARNING_RATE=0.00005 --MAX_POST_LENGTH=128 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM_V' --DO_TRAIN  \
    --MODELNAME='alt_ModelF3' --NAME='exp11' \
    --WEIGHTED_STANCE

# exp10
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=20 --BATCH_SIZE_TEST=20 --LOG_INTERVAL=10 --N_EPOCHS=8 \
    --LEARNING_RATE=0.00005 --MAX_POST_LENGTH=128 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM_V' --DO_TRAIN  \
    --MODELNAME='alt_ModelF2' --NAME='exp10' \
    --WEIGHTED_STANCE

# exp09
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=20 --BATCH_SIZE_TEST=20 --LOG_INTERVAL=10 --N_EPOCHS=8 \
    --LEARNING_RATE=0.00005 --MAX_POST_LENGTH=128 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM_V' --DO_TRAIN  \
    --MODELNAME='alt_ModelF1' --NAME='exp09' \
    --WEIGHTED_STANCE


# exp08
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=10 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF4' --NAME='exp08' \
#    --WEIGHTED_STANCE
    
# exp07
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=10 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF3' --NAME='exp07' \
#    --WEIGHTED_STANCE
        
# exp06
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=10 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF2' --NAME='exp06' \
#    --WEIGHTED_STANCE

# exp05
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=10 \
    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
    --OPTIMIZER='ADAM' --DO_TRAIN  \
    --MODELNAME='alt_ModelF1' --NAME='exp05' \
    --WEIGHTED_STANCE



# exp04
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=8 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF4' --NAME='exp04' \
#    --WEIGHTED_STANCE
    
# exp03
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=8 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF3' --NAME='exp03' \
#    --WEIGHTED_STANCE
        
# exp02
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=8 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF2' --NAME='exp02' \
#    --WEIGHTED_STANCE

# exp01
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_appr3.py \
#    --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 --N_EPOCHS=8 \
#    --LEARNING_RATE=0.00001 --MAX_POST_LENGTH=256 --STRIDES=2 --THREAD_LENGTH_DIVIDER=16 \
#    --OPTIMIZER='ADAM' --DO_TRAIN  \
#    --MODELNAME='alt_ModelF1' --NAME='exp01' \
#    --WEIGHTED_STANCE
    
#!/usr/bin/env bash
#@author: jakeyap on 20201013 1130pm

#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=5,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=6 --BATCH_SIZE_TEST=6 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelC3' --NAME='exp21' --DO_TRAIN --WEIGHTED_STANCE
# exp22
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelC2' --NAME='exp22' --DO_TRAIN --WEIGHTED_STANCE
# exp23
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelC1' --NAME='exp23' --DO_TRAIN --WEIGHTED_STANCE

# exp28
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,4,5,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelB4' --NAME='exp28' --DO_TRAIN --WEIGHTED_STANCE --DOUBLESTEP
# exp29
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,4,5,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelB3' --NAME='exp29' --DO_TRAIN --WEIGHTED_STANCE --DOUBLESTEP
# exp30
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,4,5,7 python main.py --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelB2' --NAME='exp30' --DO_TRAIN --WEIGHTED_STANCE --DOUBLESTEP

# exp31
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,5,6,7 python main_deep.py \
    #--N_EPOCHS=4 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    #--LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD4' \
    #--NAME='exp31' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    #--MAX_POST_PER_THREAD=4 --EXPOSED_POSTS=4
    
# exp36
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,5,6,7 python main_deep.py \
    #--N_EPOCHS=4 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    #--LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD3' \
    #--NAME='exp36' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    #--MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp37
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,5,6,7 python main_deep.py \
    #--N_EPOCHS=4 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    #--LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD2' \
    #--NAME='exp37' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    #--MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp38
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,5,6,7 python main_deep.py \
    #--N_EPOCHS=4 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    #--LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD1' \
    #--NAME='exp38' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    #--MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4

# on quadro server    
# exp43
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,2 python main_deep.py \
#    --N_EPOCHS=4 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD4' \
#    --NAME='exp43' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp44
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,2 python main_deep.py \
#    --N_EPOCHS=4 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD3' \
#    --NAME='exp44' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp45
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,2 python main_deep.py \
#    --N_EPOCHS=4 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD2' \
#    --NAME='exp45' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp46
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3,2 python main_deep.py \
#    --N_EPOCHS=4 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=12 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD1' \
#    --NAME='exp46' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp47
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=6 --BATCH_SIZE_TEST=6 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD1' \
#    --NAME='exp47' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp48
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=2,3 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=12 --BATCH_SIZE_TEST=6 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelD2' \
#    --NAME='exp48' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp49
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE2' \
#    --NAME='exp49' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
    
# exp50
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=3 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE1' \
#    --NAME='exp50' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4

# exp51
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE2' \
#    --NAME='exp51' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp52
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
#    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
#    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE1' \
#    --NAME='exp52' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
#    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP

# exp53
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE4' \
    --NAME='exp53' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp54
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE3' \
    --NAME='exp54' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP

# exp55
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00002 --OPTIMIZER='ADAM' --MODELNAME='ModelE4' \
    --NAME='exp55' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp56
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00002 --OPTIMIZER='ADAM' --MODELNAME='ModelE3' \
    --NAME='exp56' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp57
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00002 --OPTIMIZER='ADAM' --MODELNAME='ModelE4' \
    --NAME='exp57' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
    
# exp58
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1 python main_deep.py \
    --N_EPOCHS=10 --BATCH_SIZE_TRAIN=8 --BATCH_SIZE_TEST=8 --LOG_INTERVAL=10 \
    --LEARNING_RATE=0.00002 --OPTIMIZER='ADAM' --MODELNAME='ModelE3' \
    --NAME='exp58' --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4 --DOUBLESTEP
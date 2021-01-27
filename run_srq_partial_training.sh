#!/usr/bin/env bash
#@author: jakeyap on 20201025 10:00pm

# this is to train a mapping layer from 10 Coase Discourse classes to 4 SDQC classes
#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python tests_on_srq_addon.py --MODELFILE='alt_ModelG3_exp31.bin' \
#    --BATCH_SIZE_TRAIN=2 --BATCH_SIZE_TEST=96 --N_EPOCHS=10 --LEARNING_RATE=0.0005 \
#    --MAX_POST_LENGTH=256 --MAX_POST_PER_THREAD=7 --MAP_OPTION=2 --LOG_INTERVAL=1 --DO_TRAIN

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python tests_on_srq_addon.py --MODELFILE='alt_ModelF4_exp24.bin' \
    --BATCH_SIZE_TEST=20 --MAX_POST_LENGTH=128 --MAX_POST_PER_THREAD=7 --N_EPOCHS=5 --MAP_OPTION=2 --LOG_INTERVAL=10 --TRAIN_PCT=20 --DO_TRAIN 
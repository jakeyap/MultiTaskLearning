#!/usr/bin/env bash
#@author: jakeyap on 20201014 3:00pm

#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --N_EPOCHS=2 --BATCH_SIZE_TRAIN=25 --LOG_INTERVAL=1 --LEARNING_RATE=0.00005 --MOMENTUM=0.125 --OPTIMIZER='SGD' --DEBUG --WEIGHTED_STANCE

#PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py --N_EPOCHS=2 --BATCH_SIZE_TRAIN=25 --LOG_INTERVAL=1 --LEARNING_RATE=0.00005 --MOMENTUM=0.125 --OPTIMIZER='ADAM' --DEBUG --WEIGHTED_STANCE

#!/usr/bin/env bash
#@author: jakeyap on 20201013 1130pm

#conda activate env1

PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python main_deep.py --N_EPOCHS=10 \
    --BATCH_SIZE_TRAIN=1 --BATCH_SIZE_TEST=2 --LOG_INTERVAL=1 \
    --LEARNING_RATE=0.00001 --OPTIMIZER='ADAM' --MODELNAME='ModelE2' \
    --NAME='test' --DEBUG --DO_TRAIN --WEIGHTED_STANCE --THREAD_LENGTH_DIVIDER=8 \
    --MAX_POST_PER_THREAD=8 --EXPOSED_POSTS=4
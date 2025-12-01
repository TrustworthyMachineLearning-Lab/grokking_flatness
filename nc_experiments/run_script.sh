#!/bin/bash

readonly LEARNING_RATE=1e-02
readonly DATE=$(date +%m%d)
readonly SEED=15213
readonly TRAINING_EPOCH=250
readonly CANCEL_EPOCH=10000 
readonly BSZ=64 
readonly WEIGHTDECAY=0
readonly USE_REGR=1 # using regularizer
readonly USE_POW=1 # using NCC regularizer
readonly HECOEFF=1e-3 # coefficient: lambda
readonly CLIPPING_VALUE=0 
readonly OPTM="sgd" # sgd or adamw
readonly TRAINING_SIZE=0.0 # if 0 = use all training samples. 




nohup python -u train.py \
    --learning_rate=${LEARNING_RATE} \
    --training_date=${DATE} \
    --random_seed=${SEED} \
    --epochs=${TRAINING_EPOCH} \
    --cancel_epoch=${CANCEL_EPOCH} \
    --batch_size=${BSZ} \
    --weight_decay=${WEIGHTDECAY} \
    --use_regulation=${USE_REGR} \
    --use_pow=${USE_POW} \
    --lambda=${HECOEFF} \
    --clip_value=${CLIPPING_VALUE} \
    --optim_type=${OPTM} \
    --training_size=${TRAINING_SIZE} \
    > Seed${SEED}_Date${DATE}_TSZ${TRAINING_SIZE}_BSZ${BSZ}_OPTM${OPTM}_LR${LEARNING_RATE}_EPOCH${TRAINING_EPOCH}_CANECH${CANCEL_EPOCH}_WD${WEIGHTDECAY}_REGR${USE_REGR}_LIPZ${USE_POW}_LAMBDA${HECOEFF}_GCP${CLIPPING_VALUE}_trainlog.txt 2>&1 &

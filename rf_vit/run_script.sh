#!/bin/bash

readonly LEARNING_RATE=1e-02
readonly DATE=$(date +%m%d)
readonly SEED=42
readonly TRAINING_EPOCH=300
readonly CANCEL_EPOCH=150 # larger than the training epoch indicating no cancallation.
readonly BUMP_X=5.0
readonly BSZ=256 # batch size
readonly WEIGHTDECAY=0
readonly NWORKER=2
readonly PRETRAINED=0
readonly USE_REGR=1 # using regularizer
readonly HECOEFF=1e-2 # coefficient: lambda
readonly TAU=2.0
readonly WEIGHT_CAP=1
readonly CAP_TYPE="frob" # frob or spectral
readonly PHI_NORM=1
readonly F_MAX_NORM=150.0
readonly MODEL_TYPE="tiny"


nohup python -u train_vit.py \
    --lr=${LEARNING_RATE} \
    --model_type=${MODEL_TYPE} \
    --training_date=${DATE} \
    --random_seed=${SEED} \
    --epochs=${TRAINING_EPOCH} \
    --tau=${TAU} \
    --weight_cap=${WEIGHT_CAP} \
    --bump_x=${BUMP_X} \
    --cap_type=${CAP_TYPE} \
    --f_max_norm=${F_MAX_NORM} \
    --cancel_epoch=${CANCEL_EPOCH} \
    --batch_size=${BSZ} \
    --pretrained=${PRETRAINED} \
    --weight_decay=${WEIGHTDECAY} \
    --use_regulation=${USE_REGR} \
    --num_workers=${NWORKER} \
    --phi_norm=${PHI_NORM} \
    --cof_lambda=${HECOEFF} \
    > Seed${SEED}_Date${DATE}_BSZ${BSZ}_PRETRAINED${PRETRAINED}_MD${MODEL_TYPE}_LR${LEARNING_RATE}_EPOCH${TRAINING_EPOCH}_CANECH${CANCEL_EPOCH}_BUMP${BUMP_X}_WD${WEIGHTDECAY}_REGR${USE_REGR}_PHIN${PHI_NORM}_LAMBDA${HECOEFF}_TAU${TAU}_WCAP${WEIGHT_CAP}_CAPT${CAP_TYPE}_FMAX${F_MAX_NORM}_trainlog.txt 2>&1 &

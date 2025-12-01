#!/bin/bash

readonly SEED=42
readonly WEIGHTDECAY=1.0
readonly DATE=$(date +%m%d)

readonly TRAIN_DATA=20.0
readonly ACTUAL_TRAIN_DATA=20.0
readonly VAL_DATA=80.0

readonly LEARNING_RATE=1e-4

readonly OPTIM="AdamW" # optimizer either sgd or AdamW

readonly CLIPPING_VALUE=0.0 # clip when using regularization.

nohup python -u scripts/train.py \
    --random_seed=${SEED} \
    --train_data_pct=${TRAIN_DATA} \
    --actual_train_data_pct=${ACTUAL_TRAIN_DATA} \
    --hessian_coeff=${HECOEFF} \
    --hessian_coeff_direction=${HECOEFF_SIGN} \
    --weight_decay=${WEIGHTDECAY} \
    --use_rglr=${USE_REGR} \
    --reg_step=${REGR_STEP} \
    --use_pow=${USE_POW} \
    --freeze=${USE_FREEZE} \
    --max_hessian_coeff=${MAX_HECOEFF} \
    --use_schedular=${USE_SCHEDULAR} \
    --max_lr=${LEARNING_RATE} \
    --clip_value=${CLIPPING_VALUE} \
    --opt_type=${OPTIM} \
    --training_date=${DATE}_${LEARNING_RATE}LR_${OPTIM}OPM_${WEIGHTDECAY}WD_${HECOEFF}COEF_${HECOEFF_SIGN}COEFSN_${USE_REGR}REGR_${USE_POW}POW_${REGR_STEP}STEP_${USE_FREEZE}FRZ_${CLIPPING_VALUE}GCP_${SCDLR}SCDLR_${MAX_HECOEFF}MAXCOEF \
    --max_steps=100000 > seed${SEED}_train${ACTUAL_TRAIN_DATA}_${DATE}_${LEARNING_RATE}LR_${OPTIM}OPM_${WEIGHTDECAY}WD_${HECOEFF}COEF_${HECOEFF_SIGN}COEFSN_${USE_REGR}REGR_${USE_POW}POW_${REGR_STEP}STEP_${USE_FREEZE}FRZ_${CLIPPING_VALUE}GCP_${SCDLR}SCDLR_${MAX_HECOEFF}MAXCOEF.txt 2>&1 &

#!/bin/bash

readonly LEARNING_RATE=1e-3
readonly DP=0.0
readonly DATE=$(date +%m%d)
readonly SEED=42
readonly TRAINING_EPOCH=50
readonly CANCEL_EPOCH=2000
readonly BSZ=32
readonly WEIGHTDECAY=0
readonly USE_REGR=0
readonly HECOEFF=0
readonly TAU=0
readonly WEIGHT_CAP=0
readonly CAP_TYPE="None" # frob or spectral
readonly PHI_NORM=0
readonly F_MAX_NORM=0
readonly HEAD_ONLY=0
readonly TASK="sst5"
readonly TSZ=0
readonly BUMPX=0


nohup python -u bert_sst5.py \
    --lr=${LEARNING_RATE} \
    --task=${TASK} \
    --head_only=${HEAD_ONLY} \
    --dropout=${DP} \
    --training_date=${DATE} \
    --seed=${SEED} \
    --epochs=${TRAINING_EPOCH} \
    --cancel_epoch=${CANCEL_EPOCH} \
    --bump_x=${BUMPX} \
    --tau=${TAU} \
    --train_subset=${TSZ} \
    --weight_cap=${WEIGHT_CAP} \
    --cap_type=${CAP_TYPE} \
    --f_max_norm=${F_MAX_NORM} \
    --batch_size=${BSZ} \
    --weight_decay=${WEIGHTDECAY} \
    --use_regulation=${USE_REGR} \
    --phi_norm=${PHI_NORM} \
    --lmbd=${HECOEFF} \
    > BERT_${TASK}_Seed${SEED}_Date${DATE}_EPOCH${TRAINING_EPOCH}_CANCEL${CANCEL_EPOCH}_BUMPX${BUMPX}_BSZ${BSZ}_TSZ${TSZ}_LR${LEARNING_RATE}_HD${HEAD_ONLY}_DP${DP}_WD${WEIGHTDECAY}_REGR${USE_REGR}_PHIN${PHI_NORM}_LMBD${HECOEFF}_TAU${TAU}_WCAP${WEIGHT_CAP}_CAPT${CAP_TYPE}_FMAX${F_MAX_NORM}_trainlog.txt 2>&1 &


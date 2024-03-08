#!/bin/bash

DATASET_URL=/media/gustavo/Storage/Datasets/obfuscation
ATTACKER_URL=/home/gustavo/gitlab/cg_branch
TRAIN_REC_URL=/home/gustavo/gitlab/input-perturbation
conda activate recbole
export CUDA_VISIBLE_DEVICES="0"
DATASET=$1
OUT_DATA=/media/gustavo/Storage/obf_dummy/all_small_test/
CUDA_LAUNCH_BLOCKING=1 python $ATTACKER_URL/src/simple_attack.py  \
 --atk_config attacker_configs/"$DATASET"_gender_weighted.yaml \
 --dataset $DATASET --custom_out_dir $OUT_DATA --gpus "0" --n_parallel 2  --n_folds 5
#!/bin/bash

DATASET_URL=/share/hel/datasets/cf_obfuscation
ATTACKER_URL=/home/gustavoe/gitlab/attacker_code
TRAIN_REC_URL=/home/gustavoe/gitlab/recsys_input_perturbation
conda activate recbole-bert
export CUDA_VISIBLE_DEVICES="0,1"
DATASET=$1
OUT_DATA=/home/gustavoe/all_full_test_50/
python $ATTACKER_URL/src/simple_attack.py  \
 --atk_config attacker_configs/"$DATASET"_gender_weighted.yaml \
 --dataset $DATASET --custom_out_dir $OUT_DATA --gpus "0,1" --n_parallel 6  --n_folds 5
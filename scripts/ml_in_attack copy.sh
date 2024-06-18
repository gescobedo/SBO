#!/bin/bash

DATASET_URL=/share/hel/datasets/cf_obfuscation
ATTACKER_URL=/home/gustavoe/gitlab/attacker_code
TRAIN_REC_URL=/home/gustavoe/gitlab/recsys_input_perturbation
CONFIG_URL=/home/gustavoe/gitlab/scripts 
conda activate recbole-bert

DATASET=$1
ALG=$2
OUT_DATA=$3/$(date '+%Y-%m-%d_%H%M')/
python $TRAIN_REC_URL/pool_recs.py --dataset $DATASET --model $ALG \
        --data_path $DATASET_URL --out_dir $OUT_DATA --nproc 2 --gpu 2

python $ATTACKER_URL/src/simple_attack.py  \
 --atk_config $CONFIG_URL/attacker_configs/ml-1m_gender_weighted.yaml \
 --dataset $DATASET --custom_out_dir $OUT_DATA --gpus "0" --n_parallel 3  --n_folds 5
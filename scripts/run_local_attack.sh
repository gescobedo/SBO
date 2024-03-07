#!/bin/bash

DATASET_URL=/media/gustavo/Storage/Datasets/obfuscation
ATTACKER_URL=/home/gustavo/gitlab/cg_branch
TRAIN_REC_URL=/home/gustavo/gitlab/input-perturbation

conda activate recbole-alter

DATASET=$1
ALG=$2
OUT_DATA=$3/$(date '+%Y-%m-%d_%H%M')/
python $TRAIN_REC_URL/pool_recs.py --dataset $DATASET --model $ALG \
        --data_path $DATASET_URL --out_dir $OUT_DATA --nproc 2 --gpu 2
conda activate recbole
python $ATTACKER_URL/src/simple_attack.py  \
 --atk_config $ATTACKER_URL/configs/vae/bac-ml-1m-atk.yaml \
 --dataset $DATASET --custom_out_dir $OUT_DATA --gpus "0" --n_parallel 3  --n_folds 5
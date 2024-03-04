#!/bin/bash

DATASET_URL=/media/gustavo/Storage/Datasets/obfuscation/
OUT_DATA=data/
conda activate recbole-alter
export CUDA_VISIBLE_DEVICES=0
python pool_recs.py --dataset ml-1m-1000 --model BPR --data_path $DATASET_URL \
                    --out_dir $OUT_DATA --nproc 2 --gpu 2 --wandb False
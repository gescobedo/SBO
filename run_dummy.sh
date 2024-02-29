#!/bin/bash
conda activate recbole-bert



tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --dataset ml-1m-1000 --model BPR --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/dummy --nproc 1 --gpu 2'
  
  

#!/bin/bash
conda activate recbole-bert


tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --dataset lfm-100k-1000 --model BPR --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/lfm_small/ --nproc 5 --gpu 1'
    
tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --dataset lfm-100k-1000 --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/lfm_small/ --nproc 2 --gpu 2'
  
tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --dataset lfm-100k-1000 --model MultiVAE --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/lfm_small/ --nproc 2 --gpu 3'
  


tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --dataset ml-1m-1000 --model BPR --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/ml_small/ --nproc 5 --gpu 1'
    
tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --dataset ml-1m-1000 --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/ml_small/ --nproc 3 --gpu 2'
  
tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --dataset ml-1m-1000 --model MultiVAE --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/ml_small/ --nproc 2 --gpu 3'
  

#!/bin/bash
conda activate recbole-bert

#python pool_recs.py --model BPR --data_path /media/gustavo/Storage/Datasets/obfuscation/ --out_dir sample_test/ 
tmux new 'export CUDA_VISIBLE_DEVICES=1 ;. ml_train_attack.sh ml-1m-1000 BPR '
    
#tmux new 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/ --nproc 3 --gpu 2'
  
#tmux new 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model MultiVAE --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/ --nproc 2 --gpu 3'
  

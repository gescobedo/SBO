#!/bin/bash
conda activate recbole-bert

#python pool_recs.py --model BPR --data_path /media/gustavo/Storage/Datasets/obfuscation/ --out_dir sample_test/ 
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model BPR --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 9 --gpu 0'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model BPR --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 7 --gpu 1'
    
  
tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model MultiVAE --dataset ml-1m  --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 5 --gpu 3'
  
tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model MultiVAE --dataset lfm-100k  --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 2 --gpu 2'


#tmux new 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/ --nproc 3 --gpu 2'

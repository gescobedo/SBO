#!/bin/bash
conda activate recbole-bert

#python pool_recs.py --model BPR --data_path /media/gustavo/Storage/Datasets/obfuscation/ --out_dir sample_test/ 
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model BPR --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 9 --gpu 0'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model BPR --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 7 --gpu 1'
    
  
#tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model MultiVAE --dataset ml-1m  --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 5 --gpu 3'
  
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model MultiVAE --dataset lfm-100k  --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/full_train/ --nproc 2 --gpu 2'


#tmux new 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/ --nproc 3 --gpu 2'


# LFM  MULTVAE
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model MultiVAE --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.part0.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model MultiVAE --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.part1.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model MultiVAE --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.part2.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model MultiVAE --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.part3.json --nproc 2 --gpu 1'

# ML1m MULiTVAE
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model MultiVAE --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.part0.json --nproc 4 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model MultiVAE --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.part1.json --nproc 4 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model MultiVAE --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.part2.json --nproc 4 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model MultiVAE --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.part3.json --nproc 4 --gpu 1'

# ligtGCN originals
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.original.part0.json --nproc 1 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.original.part0.json --nproc 1 --gpu 1'

# ML1m lightGCN imputate
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.imputate.part0.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.imputate.part1.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.imputate.part2.json --nproc 2 --gpu 1'

# ML1m lightGCN weighted
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.weighted.part0.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.weighted.part1.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model LightGCN --dataset ml-1m --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/ml-1m/datasets.weighted.part2.json --nproc 2 --gpu 1'


#PENDING
# LFM lightGCN weighted
#tmux new -d 'export CUDA_VISIBLE_DEVICES=0 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.weighted.part0.json --nproc 2 --gpu 1'
tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.weighted.part1.json --nproc 2 --gpu 1'
tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.weighted.part2.json --nproc 2 --gpu 1'

# LFM lightGCN imputate
tmux new -d 'export CUDA_VISIBLE_DEVICES=3 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.imputate.part0.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=1 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.imputate.part1.json --nproc 2 --gpu 1'
#tmux new -d 'export CUDA_VISIBLE_DEVICES=2 ; python pool_recs.py --model LightGCN --dataset lfm-100k --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/all_test_mean_stereo/ --datasets_file /home/gustavoe/all_test_mean_stereo/lfm-100k/datasets.imputate.part2.json --nproc 2 --gpu 1'

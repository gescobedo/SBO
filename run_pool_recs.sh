#!/bin/bash
conda activate recbole-bert

python pool_recs.py --model BPR --data_path /share/hel/datasets/lfm-sessions-sample/ --out_dir sample_test/ 
python pool_recs.py --model NeuMF --data_path /share/hel/datasets/lfm-sessions-sample/ --out_dir sample_tests/ 
python pool_recs.py --model LightGCN --data_path /share/hel/datasets/lfm-sessions-sample/ --out_dir sample_test/ 

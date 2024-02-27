#!/bin/bash
conda activate recbole-bert

#python pool_recs.py --model BPR --data_path /media/gustavo/Storage/Datasets/obfuscation/ --out_dir sample_test/ 

python pool_recs.py --model BPR --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/ 
python pool_recs.py --model MultiVAE --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/  
python pool_recs.py --model LightGCN --data_path /share/hel/datasets/cf_obfuscation/ --out_dir /home/gustavoe/obfuscation/sample_test/  

#!/bin/bash
conda activate sbo

# Setting up directories 
ATTACKER_EXEC_URL=../attacker_code # Attacker network executable
ATTACKER_RESULT_FILES_URL=../results # Attacker results files
DATASET_URL=../Datasets  # Replace this with <ROOT_DIR_STR> in constants.py


DATASET_CONFIG=$1 # select the available attacker configurations [ml-1m, lfm-100k, ml-1m-1000, lfm-100k-1000]
DATASETS_FILE=$2 # for testing../datasets_files/datasets_sample.json list of obfuscated datasets to train attackers and recommenders

# Attacker 
python $ATTACKER_EXEC_URL/src/simple_attack.py  \
 --atk_config attacker_configs/"$DATASET_CONFIG"_gender_weighted.yaml \
 --data_path $DATASET_URL --datasets_file $DATASETS_FILE  --atk_results_dir $ATTACKER_RESULT_FILES_URL \
 --gpus "0,1" --n_parallel 2  --n_folds 5



# Template
# cd scripts
# sh run_train_attack.sh <DATASET_CONFIG> <DATASETS_FILES> 

# Example
# sh run_train_attack.sh ml-1m ../datasets_files/datasets_sample.json




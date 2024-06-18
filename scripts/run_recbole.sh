#!/bin/bash
conda activate sbo

# Setting up directories 
RECBOLE_RESULTS_FILES_URL=../recbole-results #Recommender models results files

DATASET_URL=../Datasets  # Replace this with <ROOT_DIR_STR> in constants.py

DATASETS_FILE=$1 # for testing../datasets_files/datasets_sample.json list of obfuscated datasets to train attackers and recommenders
RECOMMENDER_MODEL=$2 # Recbole supported model in  [BPR, MultiVAE, LightGCN]

# Recommender Model
python pool_recs.py --model $RECOMMENDER_MODEL --data_path $DATASET_URL \
    --datasets_file $DATASETS_FILE --out_dir $RECBOLE_RESULTS_FILES_URL --nproc 2

# Template
# cd scripts
# sh run_recbole.sh <DATASETS_FILES> <RECOMMENDER_MODEL> 

# Example
# sh run_recbole.sh ../datasets_files/datasets_sample.json BPR



# Making Alice Appear Like Bob: A Probabilistic   Preference Obfuscation Method For Implicit Feedback Recommendation Models

This repository contains the source code for paper:
> Escobedo, G., Moscati, M., Müllner, P., Kopeinik, S., Kowald, D., Lex, E., and Schedl M. Making Alice Appear Like Bob: A Probabilistic Preference Obfuscation Method For Implicit Feedback Recommendation Models, Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2024), Vilnius, Lithuania, September 2024.
<!-- ```bibtex
@inproceedings{Escobedo2024SBO,
    author = "Escobedo, Gustavo and Moscati, Marta and Mullner, Peter and Kopeinik, Simone and Kowald, Dominik  and Lex, Elisabeth  and Schedl, Markus",
    title = "Making Alice Appear Like Bob: A Probabilistic Preference Obfuscation Method 
            For Implicit Feedback Recommendation Models",
    booktitle = "Machine Learning and Knowledge Discovery in Databases: Research Track",
    publisher = "Springer Nature Switzerland",
    address="",
    pages="--",
    year = 2024
}
``` -->
## Installation and configuration
```bash
# Create environment
conda env create -f environment.yml
# Activate environment
conda activate sbo
``` 
## Data preparation
- Set the root directory for all datasets by changing the variable the variable `ROOT_DIR_STR` in `constants.py`. The downloaded datasets should be in the folder `ROOT_DIR_STR` (i.e. `../<ROOT_DIR_STR>/ml-1m/..`). 
- Modify the  obfuscation parameters in `constants.py`, an obfuscated dataset will be generated for each combination of these parameters inside the previously set root directory. Moreover, all the datasets are in [RecBole's](https://recbole.io) [atomic format](https://recbole.io/docs/user_guide/data/atomic_files.html).   
```bash
# Pre-process and generate standard format
python preprocess.py
```
## Run
- Execute the following command to generate the obfuscated datasets this will generate several parallel processes for generating the different datasets.
```bash
# Generate the obfuscated dataset files
python run_obfuscation.py
```
- To train the attacker networks on the obfuscated datasets run/adapt the script `scripts\run_train_attack.sh`.
```bash
cd scripts
sh run_train_attack.sh <DATASET_CONFIG> <DATASETS_FILES> 

# Example
# sh run_train_attack.sh ml-1m ../datasets_files/datasets_sample.json 
```
- To train the RecBole models on the obfuscated datasets run/adapt the script `scripts\run_recbole.sh`.
The script executes `pool_recs.py` code which is meant to run multiple jobs. Several `csv` files will be generated containing test results of the trained models. Additionally, given that our datasets are RecBole ready, indivual model can be trained following the [documentation](https://recbole.io/docs/get_started/started/general.html)
```bash
cd scripts
sh run_recbole.sh <DATASETS_FILES> <RECOMMENDER_MODEL> 

# Example
# sh run_recbole.sh ../datasets_files/datasets_sample.json BPR 
```
## Acknowledgment

This research was funded in whole or in part by the FFG COMET center program, by the Austrian Science Fund (FWF): P36413, P33526, and DFH-23, and by the State of Upper Austria and the Federal Ministry of Education, Science, and Research, through grant LIT-2021-YOU-215 and LIT-2020-9-SEE-113.
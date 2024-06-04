# Making Alice Appear Like Bob: A Probabilistic   Preference Obfuscation Method For Implicit Feedback Recommendation Models

This repository contains the source code for paper:
> "Making Alice Appear Like Bob: A Probabilistic Preference Obfuscation Method For Implicit Feedback Recommendation Models" by Gustavo Escobedo, Marta Moscati, Peter Muellner, Simone Kopeinik, Dominik Kowald, Elisabeth Lex and Markus Schedl
```bibtex
@inproceedings{Escobedo2024SBO,
    author = 
        "Escobedo, Gustavo and 
        Moscati, Marta and 
        M\"{u}llner, Peter and 
        Kopeinik, Simone and 
        Dominik Kowald and 
        Elisabeth Lex and 
        Markus Schedl",
    title = "Making Alice Appear Like Bob: A Probabilistic Preference Obfuscation Method 
            For Implicit Feedback Recommendation Models",
    booktitle = "Machine Learning and Knowledge Discovery in Databases: Research Track",
    publisher = "Springer Nature Switzerland",
    address="",
    pages="--",
    year = 2024,
}
```
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

## Acknowledgment

This research was funded in whole or in part by the FFG COMET center program, by the Austrian Science Fund (FWF): P36413, P33526, and DFH-23, and by the State of Upper Austria and the Federal Ministry of Education, Science, and Research, through grant LIT-2021-YOU-215.
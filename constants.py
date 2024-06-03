from pathlib import Path

RAND_SEED = 42
ROOT_DIR_STR = "../Datasets" # Replace this line to path_to_datasets/Datasets 
ROOT_DIR = Path(ROOT_DIR_STR)

K_CORE = 5

OBF_METHODS=[
    "remove",
    "imputate",
    "weighted",
    ]
SAMPLE_METHODS=[
    "ff",
    "random",
    "topk"
    ]
STEREO_TYPES = [
    #"mean", 
    #"median",
    "mean-abs", 
    "median-abs",
    #"diff",
    ]
USER_STEREO_THRES= [0.5, 0.25, 0.1, 0.005, 0.0025]
USER_STEREO_THRES_DICT= {
    "ml-1m":0.30, 
    "lfm-100k":0.39,
    "ml-1m-1000":0.30, 
    "lfm-100k-1000":0.39}
P_SAMPLE = [0.05, 0.10] #,0.15]
TOPK=[50, 100]

from pathlib import Path

ROOT_DIR_STR = "/media/gustavo/Storage/Datasets"
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
    "median",
    "mean-abs", 
    "median-abs",
    "diff",
    ]
USER_STEREO_THRES= [0.5, 0.25, 0.1, 0.005, 0.0025]
P_SAMPLE = [0.05, 0.10] #,0.15]
TOPK=[50, 100]

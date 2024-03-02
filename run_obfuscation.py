from obfuscation import run_obfuscation 
from constants import *

obf_methods=[
    "remove",
    "imputate",
    #"weighted",
    ]
sample_methods=[
    "ff",
    "random",
    "topk"
    ]
stereo_types = [
    "mean", 
    "median",
    "inc_ratio",
    ]
user_stereo_thres= [0.5, 0.25, 0.1, 0.005, 0.0025]
p_sample = [0.05, 0.10, 0.15]
topk=[50, 100]

datasets_source = [
    #ROOT_DIR_STR+"/obfuscation/ml-1m",
    #ROOT_DIR_STR+"/obfuscation/lfm-100k",
    ROOT_DIR_STR+"/obfuscation/ml-1m-1000",
    #ROOT_DIR_STR+"/obfuscation/lfm-100k-1000",
    ]
obf_params=[]
for stereo_type in stereo_types:
    for data_dir in datasets_source: 
        for psample in p_sample:
            obf_params.append(
                {
                    "root_dir":ROOT_DIR_STR+"/obfuscation",
                    "data_dir":data_dir,
                    "p_sample": psample,
                    "topk":50,
                    "obf_method":"remove",
                    "sample_method":"ff",
                    "stereo_type":stereo_type,
                    "user_stereo_pref_thresh":0.005,
                }
                )


if __name__ == "__main__":
    folders = []
    for params in  obf_params:
        train,valid,test, name, folder = run_obfuscation(**params)
        folders.append(name)

    print(folders)








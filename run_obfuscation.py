from obfuscation import run_obfuscation 
from constants import *


datasets_source = [
    #ROOT_DIR_STR+"/obfuscation/ml-1m",
    #ROOT_DIR_STR+"/obfuscation/lfm-100k",
    ROOT_DIR_STR+"/obfuscation/ml-1m-1000",
    #ROOT_DIR_STR+"/obfuscation/lfm-100k-1000",
    ]
obf_params=[]
for stereo_type in STEREO_TYPES:
    for data_dir in datasets_source: 
        for psample in P_SAMPLE:
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








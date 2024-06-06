from obfuscation import run_obfuscation
from constants import *
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

datasets_source = {
    "ml-1m": ROOT_DIR_STR + "/obfuscation/ml-1m",
    "lfm-100k": ROOT_DIR_STR + "/obfuscation/lfm-100k",
    # Generating small datasets for developing
    "ml-1m-1000": ROOT_DIR_STR + "/obfuscation/ml-1m-1000",
    "lfm-100k-1000": ROOT_DIR_STR + "/obfuscation/lfm-100k-1000",
}

obf_params = []
for sample_method in SAMPLE_METHODS:
    for obf_method in OBF_METHODS:
        for stereo_type in STEREO_TYPES:
            for dataset_name, data_dir in datasets_source.items():
                for psample in P_SAMPLE:
                    obf_params.append(
                        {
                            "root_dir": ROOT_DIR_STR + "/obfuscation",
                            "data_dir": data_dir,
                            "p_sample": psample,
                            "topk": 50,
                            "obf_method": obf_method,
                            "sample_method": sample_method,
                            "stereo_type": stereo_type,
                            "user_stereo_pref_thresh": USER_STEREO_THRES_DICT[
                                dataset_name
                            ],
                            "weights": [0.5, 0.5],
                        }
                    )


if __name__ == "__main__":

    np.random.seed(RAND_SEED)
    folders = []
    results = Parallel(n_jobs=8, verbose=11)(
        delayed(run_obfuscation)(**p)
        for p in tqdm(obf_params, desc="Generating Datasets", position=0, leave=True)
    )
    print(
        [
            out_file
            for (
                train_data_obf,
                valid_data_obf,
                test_data,
                out_file,
                out_dir,
            ) in results
        ]
    )

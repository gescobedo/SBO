# %%
import pandas as pd


results_files = [
    "random_completion-1000/test_BPR_2023-08-30 16:25:46.145796.pkl",
    "random_completion-1000/test_BPR_2023-08-31 17:21:07.443649.pkl",
    "random_completion-500/test_BPR_2023-08-31 00:48:08.670203.pkl",
    "random_completion-500/test_BPR_2023-08-31 23:54:39.085208.pkl",
    "true_negatives_shrink/test_BPR_2023-08-30 10:40:29.322245.pkl",
    "true_negatives_shrink/test_BPR_2023-08-31 14:46:58.805373.pkl",
    "random_completion-10000/test_BPR_2023-09-04 14:15:10.523891.pkl",
]

results_files = [
    "random_completion-10000/test_BPR_2023-09-04 14:15:10.523891.pkl",
    "random_completion-10000/test_LightGCN_2023-09-07 06:22:01.612544.pkl",
    "random_completion-10000/test_NeuMF_2023-09-06 16:56:48.313282.pkl",
    "random_completion-10000/test_LightGCN_2023-09-08 17:29:35.080365.pkl",
    "random_completion-10000/test_LightGCN_2023-09-09 17:37:10.221435.pkl",
]
results_files = [
    "/home/gustavoe/obfuscation/ml_small/ml-1m-1000/test_BPR_2024-02-29 11:12:03.117474.pkl",
    "/home/gustavoe/obfuscation/lfm_small/lfm-100k-1000/test_BPR_2024-02-29 11:14:42.795227.pkl",
    "/home/gustavoe/obfuscation/ml_small/ml-1m-1000/test_LightGCN_2024-02-29 11:23:20.712729.pkl",
    "/home/gustavoe/obfuscation/lfm_small/lfm-100k-1000/test_LightGCN_2024-02-29 11:26:44.819582.pkl",
    "/home/gustavoe/obfuscation/ml_small/ml-1m-1000/test_MultiVAE_2024-02-29 11:24:28.270790.pkl",
    "/home/gustavoe/obfuscation/lfm_small/lfm-100k-1000/test_MultiVAE_2024-02-29 11:22:40.385898.pkl",
]
results_files = [
    # Resulst 2 epochs
    # "/share/rk4/home/gustavoe/obfuscation/full_train/lfm-100k/test_BPR_2024-03-10 19:22:21.434191.pkl",
    # "/share/rk4/home/gustavoe/obfuscation/full_train/ml-1m/test_BPR_2024-03-09 15:09:43.092869.pkl",
]

# ============== 100 EPOCHS ============
results_files_split = {
    "BPR_ALL": [
        "/share/rk4/home/gustavoe/obfuscation/full_train/ml-1m/test_BPR_2024-03-12 14:17:52.475587.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_BPR_2024-03-15 03:23:07.718920.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_BPR_2024-03-17 09:50:47.250257.pkl",
    ],
    "MULTVAE_ALL": [
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_MultiVAE_2024-03-19 03:57:56.902459.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_MultiVAE_2024-03-19 03:51:15.039839.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_MultiVAE_2024-03-19 02:38:58.455025.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/lfm-100k/test_MultiVAE_2024-03-19 03:49:03.467385.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/ml-1m/test_MultiVAE_2024-03-19 12:12:11.501643.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/ml-1m/test_MultiVAE_2024-03-19 12:14:22.534349.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/ml-1m/test_MultiVAE_2024-03-19 12:13:31.096683.pkl",
        "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo/ml-1m/test_MultiVAE_2024-03-19 11:43:30.072991.pkl",
    ],
    "LIGHTGCN_REMOVAL_ML-1M": [
        "/share/rk1/home/marta/obfuscation/all_test_mean_stereo/ml-1m/test_LightGCN_2024-03-18 22:42:08.663277.pkl",
        "/share/rk1/home/marta/obfuscation/all_test_mean_stereo/ml-1m/test_LightGCN_2024-03-18 22:56:09.040045.pkl",
        "/share/rk1/home/marta/obfuscation/all_test_mean_stereo/ml-1m/test_LightGCN_2024-03-18 23:13:16.421693.pkl",
    ],
}
results_files = []
[[results_files.append(x) for x in row] for row in list(results_files_split.values())]
# %%
#!find  /share/rk4/home/gustavoe/obfuscation/full_train/ -name *.pkl


# %%
DEFAULT_RESULTS_DIR = "/home/gustavoe/obfuscation"
# results_files = [f"{DEFAULT_RESULTS_DIR}/{file}" for file in results_files]
DEFAULT_RESULTS_DIR = ""
import pickle


def convert_table(file):
    for res in file:
        res.update(res["test_result"])

    df = pd.DataFrame.from_dict(file)
    print(df.columns)
    print(df.head())
    df["dataset"] = df["name"].apply(lambda x: "-".join(x.split("-")[1:]))
    converted_df = df.groupby(["Model", "dataset"])[
        [x for x in df.columns if x.endswith("20")]
    ].mean()
    std_df = df.groupby(["Model", "dataset"])[
        [x for x in df.columns if x.endswith("20")]
    ].std()
    return df, converted_df, std_df


from collections import defaultdict


def read_rebole_result_files(results_files):
    results_dict = defaultdict(lambda: [])
    for file_name in results_files:
        data = pickle.load(open(file_name, "rb"))
        key = file_name.split("/")[0]
        for data_item in data:
            results_dict[key].append(data_item)
    return results_dict


def process_results(results_files):
    processed_results_dict = {}
    dfs = []
    results_dict = read_rebole_result_files(results_files)
    for key, data in results_dict.items():
        df, conv_df, std = convert_table(data)
        df["key"] = key
        processed_results_dict[key] = conv_df
        print(key)
        print(processed_results_dict[key])
        # print(std)
        dfs.append(df)
    joined = pd.concat(dfs, axis=0, ignore_index=True)
    print(joined.head())
    return joined


def process_results_split(results_split, dir):
    for k, results_files in results_split.items():
        joined = process_results(results_files)
        joined.to_csv(f"{dir}/result_{k}.csv", index=False)


# %%
DEFAULT_RESULTS_DIR = "/share/rk4/home/gustavoe/obfuscation/all_test_mean_stereo"
joined = process_results(results_files)
joined.to_csv(f"{DEFAULT_RESULTS_DIR}/result_ALL.csv", index=False)
# %
process_results_split(results_files_split, DEFAULT_RESULTS_DIR)
# %%

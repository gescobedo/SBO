from datetime import datetime
import pickle
import os
import argparse
from recbole.quick_start import run_recbole
from joblib import delayed, Parallel
from pathlib import Path
import json
from constants import *
import pandas as pd
from collections import defaultdict



num_cores = 6

parameter_dict = {
    "field_separator": ",",
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "load_col": {"inter": ["user_id", "item_id"]},
    "save_dataset": True,
    "save_dataloaders": True,
    "embedding_size": 64,
    "epochs": 100,
    "train_batch_size": 512,
    "eval_batch_size": 2048,
    "benchmark_filename": ["train", "valid", "test"],
    "stopping_step": 10,
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision", "ItemCoverage"],
    "topk": [10, 20, 50],
    "metric_decimal_place": 6,
    "valid_metric": "NDCG@10",
    "use_gpu": True,
    "nproc": 1,
    "log_wandb": False,
}


def run_alg(args, model, dataset, config):
    # model, dataset, config =args

    data_path = config["out_dir"]
    config["checkpoint_dir"] = data_path + "saved/" + dataset + "/"
    config["dataset_save_path"] = (
        data_path + "saved/" + dataset + f"/{model}_dataset.pth"
    )
    config["dataloaders_save_path"] = (
        data_path + "saved/" + dataset + f"/{model}_dataloader.pth"
    )

    config["world_size"] = args.world_size
    config["ip"] = args.ip
    config["port"] = args.port
    config["nproc"] = args.nproc
    config["offset"] = args.group_offset

    if not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    params = {
        "model": model,
        "dataset": dataset,
        "config_file_list": None,
        "config_dict": config,
    }

    results = run_recbole(model=model, dataset=dataset, config_dict=config)
    results["name"] = f"{model}-{dataset}"

    return results

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


def read_rebole_result_files(results_files):
    results_dict = defaultdict(lambda: [])
    for file_name in results_files:
        data = pickle.load(open(file_name, "rb"))
        key = file_name.split("/")[0]
        for data_item in data:
            results_dict[key].append(data_item)
    return results_dict

def transform_recbole_results(data,key):
    results_dict = defaultdict(lambda: [])
    for data_item in data:
        results_dict[key].append(data_item)
    return results_dict

def process_results(recbole_results,key):
    processed_results_dict = {}
    dfs = []
    results_dict = transform_recbole_results(recbole_results,key)
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="opt_clustering")

    parser.add_argument("--model", required=True, help="path to Test file")
    parser.add_argument("--data_path", required=True, help="path to Test file")
    parser.add_argument("--out_dir", required=False)
    parser.add_argument("--gpu", type=int, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--datasets_file", type=str, required=False, default="NoFile")
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument(
        "--valid_latex", type=str, default="./latex/valid.tex", help="config files"
    )
    parser.add_argument(
        "--test_latex", type=str, default="./latex/test.tex", help="config files"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    args = parser.parse_args()

    if args.wandb:
        parameter_dict["log_wandb"] = True
    model = args.model
    models = args.model.split(",")
    dataset_name = args.dataset
    datasets = os.listdir(args.data_path)
    parameter_dict["data_path"] = args.data_path
    out_dir = args.out_dir + "/"  # +get_local_time()+"/"

    if args.datasets_file != "NoFile":
        with open(args.datasets_file, "rb") as dt_:
            datasets = json.load(dt_)["datasets"]

    parameter_dict["out_dir"] = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    params = []
    gpu_ids = [1, 2, 3] * 20
    gpu_id = 0

    for model in models:
        for dataset in datasets:
            config = parameter_dict.copy()
            config["gpu_id"] = args.gpu
            gpu_id += 1
            params.append((args, model, dataset, config))

    results_months = {}
    valid_result_list = []
    test_result_list = []

    results_list = []

    results_list = Parallel(n_jobs=args.nproc)(
        delayed(run_alg)(*param) for param in params
    )
    for res_dict in results_list:
        valid_res_dict = {"Model": model}
        test_res_dict = {"Model": model}
        result = res_dict.copy()
        valid_res_dict["best_valid_result"] = result["best_valid_result"]
        valid_res_dict["name"] = result["name"]
        test_res_dict["test_result"] = result["test_result"]
        test_res_dict["name"] = result["name"]

        bigger_flag = result["valid_score_bigger"]
        subset_columns = list(result["best_valid_result"].keys())
        valid_result_list.append(valid_res_dict)
        test_result_list.append(test_res_dict)

    print(f"Saving results to {out_dir}")

    joined = process_results(test_result_list, model)
    joined.to_csv(f"{out_dir}/result_{model}.csv", index=False)

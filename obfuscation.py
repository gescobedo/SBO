# %%
import numpy as np
import pandas as pd
import pickle
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from pd_utils import filter_by
import matplotlib.pyplot as plt
from data_utils import *
import json

def perform_action(user_data, sample, method="remove"):
    # print([sample,user_data])
    if method == "remove":
        user_data = np.setdiff1d(user_data, sample)
    elif method == "imputate":
        # print(user_data)
        user_data = np.unique(np.concatenate((user_data, sample)))

    else:
        raise Exception("Not implemented action method!")

    return user_data


def perform_sub_sampling(user_data, ff_values, method ="remove",sub_method="topk", k=50, p_sample=0.1):
    if method =="remove":
        if sub_method == "topk":
            ff_user_data = ff_values.loc[user_data].sort_values(ascending=False)
            size_sample = int(p_sample * len(user_data))
            if size_sample <= k:
                top_k = ff_user_data[:size_sample]
            else:
                top_k = ff_user_data[:k]
            user_data = top_k
        elif sub_method == "random":
            user_data = np.random.choice(
                user_data, int(p_sample * len(user_data)), replace=False
            )
        elif sub_method == "ff":
            user_data = np.random.choice(
                user_data, int(p_sample * len(user_data)), replace=False
            )
            # print(ff_values.loc[user_data])
            coins = np.array(
                [np.random.binomial(1, np.abs(p), 1)[0] for p in ff_values.loc[user_data].values]
            )
            # print([len(user_data),coins,np.nonzero(coins)])
            user_data = user_data[np.nonzero(coins)[0]]

        else:
            raise Exception("Not implemented sampling method!")
    
    elif method == "imputate":
        # This should only focus on unseen data there
        if sub_method == "topk":
            ff_unseen_user_data = np.setdiff1d(ff_values.index.values,user_data)
            ff_unseen_data= ff_values.loc[ff_unseen_user_data].sort_values(ascending=False)
            size_sample = int(p_sample * len(user_data))
            if size_sample <= k:
                top_k = ff_unseen_data[:size_sample]
            else:
                top_k = ff_unseen_data[:k]
            user_data = top_k
        elif sub_method == "random":
            unseen_user_data = np.setdiff1d(ff_values.index.values,user_data)
            user_data = np.random.choice(
                unseen_user_data, int(p_sample * len(user_data)), replace=False
            )
        elif sub_method == "ff":
            user_data = np.random.choice(
                user_data, int(p_sample * len(user_data)), replace=False
            )
            # print(ff_values.loc[user_data])
            coins = np.array(
                [np.random.binomial(1, np.abs(p), 1)[0] for p in ff_values.loc[user_data]]
            )
            # print([len(user_data),coins,np.nonzero(coins)])
            user_data = user_data[np.nonzero(coins)[0]]

        else:
            raise Exception("Not implemented sampling method!")

    return user_data


def calc_user_stereotyp_pref(ff_values, method="mean"):
    user_stereo = -1.05
    if len(ff_values > 0):
        if method == "mean":
            user_stereo = np.mean(ff_values)
        elif method == "median":
            user_stereo = np.median(ff_values)
        elif method == "diff":
            user_stereo = (
                np.sum(
                    (np.where(ff_values >= 0)) - np.sum(np.abs(np.where(ff_values < 0)))
                )
                / 2
            )
        elif method == "inc_ratio":
            user_stereo = (np.sum(np.where(ff_values >= 0)) / np.sum(np.abs(np.where(ff_values < 0))))
        else:
            raise Exception("Not implemented stereotypical user preferences measure")

    return user_stereo


def obfuscate_user_data(
    user_data,
    ff_data,
    method,
    sub_method,
    topk,
    p_sample,
    sterotyp_method,
    user_stereo_pref_thresh,
):

    valid_user_items = np.intersect1d(user_data["itemID"].values, ff_data.index.values)
    user_ff_values = ff_data.loc[valid_user_items, "FF"]
    # Estimating the stereotypicallity of the user profile
    user_stereo_pref = calc_user_stereotyp_pref(
        user_ff_values.values, method=sterotyp_method
    )
    if user_stereo_pref > user_stereo_pref_thresh:
        # Sampling from user profile
        user_sampled = perform_sub_sampling(
            user_data=valid_user_items,
            ff_values=ff_data,
            method=method,
            sub_method=sub_method,
            k=topk,
            p_sample=p_sample,
        )
        # Perform obfuscation
        obfuscated_user_data = perform_action(
            valid_user_items, user_sampled, method=method
        )

        return obfuscated_user_data
    else:
        return user_data

def prepare_user_to_obf(user, train_data,ff_data,sterotyp_method):
    user_data = train_data.loc[train_data["userID"] == user]
    # Selecting only items that have defined FF values from the user profile
    valid_user_items = np.intersect1d(
        user_data["itemID"].values, ff_data.index.values
    )
    user_ff_values = ff_data.loc[valid_user_items, "FF"]
    # Estimating the stereotypicallity of the user profile
    user_stereo_pref = calc_user_stereotyp_pref(
        user_ff_values.values, method=sterotyp_method
    )
    return user_data, valid_user_items, user_ff_values, user_stereo_pref
def obfuscate_data(
    train_data,
    users,
    ff_data,
    p_SAMPLE=0.15,
    topk=50,
    method="remove",
    sub_method="ff",
    sterotyp_method="mean",
    user_stereo_pref_thresh=0.005,
):
    n_obfuscated = 0
    users_obfuscated = []
    for user in users:

        #Prepare user for obfuscation
        user_data, valid_user_items, user_ff_values, user_stereo_pref = prepare_user_to_obf(user, train_data,ff_data,sterotyp_method)

        # Sampling for users that have reached stereotypical preference threshold
        if abs(user_stereo_pref) > user_stereo_pref_thresh:
            n_obfuscated += 1
            # Sampling from user profile
            user_sampled = perform_sub_sampling(
                user_data=valid_user_items,
                ff_values=ff_data,
                sub_method=sub_method,
                k=topk,
                p_sample=p_SAMPLE,
            )
            # Perform obfuscation
            obfuscated_user_data = perform_action(
                valid_user_items, user_sampled, method=method
            )
            users_obfuscated.append([user, list(obfuscated_user_data)])
        else:
            users_obfuscated.append([user, list(user_data["itemID"])])


    obfuscated_data = pd.DataFrame(
        data=users_obfuscated, columns=["userID", "itemID"]
    ).explode("itemID",ignore_index=True)
    print(
        [
            len(train_data),
            len(obfuscated_data),
            len(users),
            p_SAMPLE,
            topk,
            method,
            sub_method,
            user_stereo_pref_thresh,
            n_obfuscated,
        ]
    )
    return obfuscated_data


def run_obfuscation(
    root_dir,
    data_dir,
    p_sample=0.15,
    topk=50,
    obf_method="remove",
    sample_method="ff",
    stereo_type="mean",
    user_stereo_pref_thresh=0.01,
):
    train_data, valid_data, test_data, inclination_data, user_features, dataset_name = (
        read_dataset_to_obfuscate(data_dir)
    )
    print(user_features.head())
    out_file = f"{dataset_name}_{obf_method}_{p_sample}_{sample_method}_{stereo_type}_th{user_stereo_pref_thresh}"
    inter_data = pd.concat([train_data, valid_data], ignore_index=True)
    obf_data = obfuscate_data(
        inter_data,
        user_features["userID"].values,
        inclination_data,
        p_SAMPLE=p_sample,
        topk=topk,
        method=obf_method,
        sub_method=sample_method,
        sterotyp_method=stereo_type,
        user_stereo_pref_thresh=user_stereo_pref_thresh,
    )
    print(f"Saving file in:\n{out_file}")

    out_dir = f"{root_dir}/{out_file}"
    if not os.path.exists(f"{root_dir}/{out_file}"):
        os.makedirs(out_dir)

    # Splitting and saving obfuscated data
    obf_data = obf_data.merge(user_features, on="userID",how="left")
    train_data_obf, valid_data_obf = split_by_inter_ratio(obf_data)
    save_recbole_data(train_data_obf,valid_data_obf,test_data,out_dir)
    config_dict={
        "p_sample":p_sample,
        "topk":topk,
        "obf_method":obf_method,
        "sample_method":sample_method,
        "stereo_type":stereo_type,
        "user_stereo_pref_thresh":user_stereo_pref_thresh,
        }
    save_csr_matrix(out_dir,obf_data)
    inclination_data.to_csv(f"{out_dir}/{out_file}_gender_incl.csv",)
    with open(f"{out_dir}/config.json", 'w') as f:
        json.dump(config_dict, f)
    print("finished obfuscation")
    #train_data_obf.to_csv(f"{out_dir}/{out_file}.train.inter", index=False)
    #valid_data_obf.to_csv(f"{out_dir}/{out_file}.valid.inter", index=False)
    #test_data.to_csv(f"{out_dir}/{out_file}.test.inter", index=False)
    return train_data_obf,valid_data_obf,test_data, out_file, out_dir

# %%


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Performing obfuscation")

    parser.add_argument(
        "--root_dir", required=True, default="data", help="path to root datasets"
    )
    parser.add_argument("--data_url", required=True, help="path to training data")
    parser.add_argument(
        "--data_incl_url",
        required=True,
        type=str,
        default=None,
        help="path to inclination data",
    )
    parser.add_argument("--p_sample", required=False, type=float, default=0.10)
    parser.add_argument("--topk", required=False, type=int, default=100)
    parser.add_argument("--obf_method", required=False, default="remove")
    parser.add_argument("--sample_method", required=False, default="topk")
    parser.add_argument("--stereo_type", required=False, default="median")
    parser.add_argument("--stereo_thres", required=False, type=float, default=0.005)

    args = parser.parse_args()

    root_dir = args.root_dir
    data_dir = args.data_url

    train_data, valid_data, test_data, inclination_data, unique_users, dataset_name = (
        read_dataset_to_obfuscate(data_dir)
    )
    out_file = f"{dataset_name}_{args.obf_method}_{args.sample_method}_{args.stereo_type}_{args.stereo_thres}"

    obf_data = obfuscate_data(
        train_data,
        unique_users,
        inclination_data,
        p_SAMPLE=args.p_sample,
        topk=args.topk,
        method=args.obf_method,
        sub_method=args.sample_method,
        sterotyp_method=args.stereo_type,
        user_stereo_pref_thresh=args.stereo_thres,
    )
    print(f"Saving file in:\n{out_file}")

    out_dir = f"{root_dir}/{out_file}"
    if not os.path.exists(f"{root_dir}/{out_file}"):
        os.makedirs(out_dir)

    # Splitting and saving obfuscated data
    train_data_obf, valid_data_obf = split_by_inter_ratio(obf_data)
    train_data_obf.to_csv(f"{out_dir}/{out_file}.train.inter", index=False)
    valid_data_obf.to_csv(f"{out_dir}/{out_file}.valid.inter", index=False)

# %%import numpy as np
import pandas as pd
import os
from pathlib import Path
from data_utils import *
from stereo_utils import *
from constants import *
from obfuscation import *


# %%
def generate_item_ster_vectors(dataset_name, root_data=ROOT_DIR_STR):
    item_mapping = pd.read_csv(
        Path(root_data) / dataset_name / "item_mapping.csv",
        names=["itemID", "token"],
        skiprows=1,
    )
    item_ster = pd.read_csv(
        Path(root_data) / dataset_name / f"{dataset_name}_gender_incl.csv",
        index_col="itemID",
    )
    merged = (
        item_mapping.merge(item_ster, on="itemID", how="left")
        .fillna(0)
        .sort_values("token")
    )
    item_ster_values = np.array(merged["FF"].values).reshape(1, -1)
    np.save(Path(root_data) / dataset_name / "item_ster_values.npy", item_ster_values)
    return item_ster_values


def generate_genre_inclination(data, out_dir, name):
    user_groups = {"gender": ["M", "F"]}
    rel_freq = calc_rel_freq_inter(data, user_groups)
    ff = calculate_ff(rel_freq, "gender_M", "gender_F")
    ff.to_csv(Path(out_dir) / f"{name}_gender_incl.csv")
    return ff


def preprocess_lfm():
    data_inter = pd.read_csv(
        ROOT_DIR / "lfm-100k/inter.tsv.bz2",
        sep="\t",
        names=["userID", "itemID", "freq"],
        engine="python",
    )
    data_user = pd.read_csv(
        ROOT_DIR / "lfm-100k/demo.tsv.bz2",
        sep="\t",
        names=["userID", "country", "age", "gender", "created_at"],
        engine="python",
    )
    joined = data_inter.merge(data_user[["userID", "gender"]], on="userID")
    joined["gender"] = joined["gender"].str.upper()
    print(joined.nunique())
    joined = core_filtering(joined, K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/lfm-100k"
    if not os.path.exists(f"{ROOT_DIR}/obfuscation/lfm-100k"):
        os.makedirs(out_dir)
    inclination_data = generate_genre_inclination(joined, out_dir, "lfm-100k")
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/lfm-100k/lfm-100k_inter.csv", index=False)
    print(joined.nunique())
    train_data, valid_data, test_data = split_by_inter_ratio_recbole(joined)
    user_dataset = transform_to_obf(
        pd.concat([train_data, valid_data], ignore_index=True)
    )
    user_stereo_scores = calc_all_user_stereotyp_pref(user_dataset, inclination_data)
    user_stereo_scores.to_csv(f"{ROOT_DIR}/obfuscation/lfm-100k/lfm-100k_user_ster.csv")
    save_recbole_data(train_data, valid_data, test_data, out_dir)
    save_csr_matrix(out_dir, transform_to_obf(joined))


def preprocess_ml1m():
    data_inter = pd.read_csv(
        ROOT_DIR / "ml-1m/ratings.dat",
        sep="::",
        names=["userID", "itemID", "rating", "timestamp"],
        engine="python",
    )
    data_user = pd.read_csv(
        ROOT_DIR / "ml-1m/users.dat",
        sep="::",
        names=["userID", "gender", "age", "occcupation", "zipcode"],
        engine="python",
    )

    joined = data_inter.merge(data_user[["userID", "gender"]], on="userID").dropna()
    print(joined.nunique())
    joined = core_filtering(joined, K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/ml-1m"
    if not os.path.exists(f"{ROOT_DIR}/obfuscation/ml-1m"):
        os.makedirs(out_dir)
    inclination_data = generate_genre_inclination(joined, out_dir, "ml-1m")
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/ml-1m/ml-1m_inter.csv", index=False)
    print(joined.nunique())
    train_data, valid_data, test_data = split_by_inter_ratio_recbole(joined)
    user_dataset = transform_to_obf(
        pd.concat([train_data, valid_data], ignore_index=True)
    )
    user_stereo_scores = calc_all_user_stereotyp_pref(user_dataset, inclination_data)
    user_stereo_scores.to_csv(f"{ROOT_DIR}/obfuscation/ml-1m/ml-1m_user_ster.csv")
    save_recbole_data(train_data, valid_data, test_data, out_dir)
    save_csr_matrix(out_dir, transform_to_obf(joined))


def core_filtering(data, min_k):
    while True:
        item_user_counts = data.groupby(["itemID"])["userID"].nunique().reset_index()
        user_item_counts = data.groupby(["userID"])["itemID"].nunique().reset_index()

        valid_items = item_user_counts[item_user_counts["userID"] >= min_k][
            "itemID"
        ].values
        valid_users = user_item_counts[user_item_counts["itemID"] >= min_k][
            "userID"
        ].values

        result = data[
            data["itemID"].isin(valid_items) & data["userID"].isin(valid_users)
        ]

        item_user_counts = (
            result.groupby(["itemID"])["userID"].nunique().reset_index()["userID"]
        )
        user_item_counts = (
            result.groupby(["userID"])["itemID"].nunique().reset_index()["itemID"]
        )

        if (item_user_counts >= min_k).any() and (user_item_counts >= min_k).any():
            break
        else:
            data = result
            print("Iterate filtering")
            if len(data) < 1000:
                print("invalid")
                break
    return result


def generate_small_lfm(n_user=1000, random_state=42):
    data_inter = pd.read_csv(
        ROOT_DIR / "lfm-100k/inter.tsv.bz2",
        sep="\t",
        names=["userID", "itemID", "freq"],
        engine="python",
    )
    data_user = pd.read_csv(
        ROOT_DIR / "lfm-100k/demo.tsv.bz2",
        sep="\t",
        names=["userID", "country", "age", "gender", "created_at"],
        engine="python",
    )
    joined = data_inter.merge(data_user[["userID", "gender"]], on="userID")
    joined["gender"] = joined["gender"].str.upper()
    sampled_users = (
        joined.groupby(["gender"])
        .apply(
            lambda x: np.random.choice(x["userID"].unique(), n_user // 2, replace=False)
        )
        .explode()
        .values
    )
    joined = joined[joined["userID"].isin(sampled_users)]
    print(joined.nunique())
    name = f"lfm-100k-{n_user}"
    joined = core_filtering(joined, K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/{name}"
    if not os.path.exists(f"{ROOT_DIR}/obfuscation/{name}"):
        os.makedirs(out_dir)
    inclination_data = generate_genre_inclination(joined, out_dir, name)
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/{name}/{name}_inter.csv", index=False)
    print(joined.nunique())
    train_data, valid_data, test_data = split_by_inter_ratio_recbole(joined)
    user_dataset = transform_to_obf(
        pd.concat([train_data, valid_data], ignore_index=True)
    )
    user_stereo_scores = calc_all_user_stereotyp_pref(user_dataset, inclination_data)
    user_stereo_scores.to_csv(f"{ROOT_DIR}/obfuscation/{name}/{name}_user_ster.csv")
    save_recbole_data(train_data, valid_data, test_data, out_dir)
    save_csr_matrix(out_dir, transform_to_obf(joined))


def generate_small_ml1m(n_user=1000, random_state=42):
    data_inter = pd.read_csv(
        ROOT_DIR / "ml-1m/ratings.dat",
        sep="::",
        names=["userID", "itemID", "rating", "timestamp"],
        engine="python",
    )
    data_user = pd.read_csv(
        ROOT_DIR / "ml-1m/users.dat",
        sep="::",
        names=["userID", "gender", "age", "occcupation", "zipcode"],
        engine="python",
    )
    joined = data_inter.merge(data_user[["userID", "gender"]], on="userID").dropna()
    joined["gender"] = joined["gender"].str.upper()
    sampled_users = (
        joined.groupby(["gender"])
        .apply(
            lambda x: np.random.choice(x["userID"].unique(), n_user // 2, replace=False)
        )
        .explode()
        .values
    )
    joined = joined[joined["userID"].isin(sampled_users)]
    print(joined.nunique())
    name = f"ml-1m-{n_user}"
    joined = core_filtering(joined, K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/{name}"
    if not os.path.exists(f"{ROOT_DIR}/obfuscation/{name}"):
        os.makedirs(out_dir)
    inclination_data = generate_genre_inclination(joined, out_dir, name)
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/{name}/{name}_inter.csv", index=False)
    print(joined.nunique())
    train_data, valid_data, test_data = split_by_inter_ratio_recbole(joined)
    user_dataset = transform_to_obf(
        pd.concat([train_data, valid_data], ignore_index=True)
    )
    user_stereo_scores = calc_all_user_stereotyp_pref(user_dataset, inclination_data)
    user_stereo_scores.to_csv(f"{ROOT_DIR}/obfuscation/{name}/{name}_user_ster.csv")
    save_recbole_data(train_data, valid_data, test_data, out_dir)
    save_csr_matrix(out_dir, transform_to_obf(joined))


if __name__ == "__main__":
    preprocess_lfm()
    preprocess_ml1m()
    generate_small_lfm()
    generate_small_ml1m()

import numpy as np
import pandas as pd
from constants import *


def item_relative_freq(data):
    # |U_i|/|U|

    rel_freq = (data.groupby("itemID")["userID"].nunique()) / data["userID"].nunique()
    rel_freq.name = "FF"
    return rel_freq


def calculate_ff(rel_freq, x, y):

    common = np.intersect1d(rel_freq[x].index, rel_freq[y].index)
    common_concat = pd.concat(
        [rel_freq[x].loc[common], rel_freq[y].loc[common]], axis=1
    )
    ff = (rel_freq[x]["FF"].loc[common] - rel_freq[y]["FF"].loc[common]) / (
        1e-24 + np.max(common_concat["FF"], axis=1)
    )
    ff.name = "FF"
    return ff


def calc_rel_freq(joined, user_groups):
    rel_freq = {}
    for group, g_values in user_groups.items():
        for g_value in g_values:
            group_data = joined.filter_by([(group, "=", g_value)])
            item_rel_freq = item_relative_freq(group_data)
            item_list = pd.Series(
                index=joined["itemID"].unique(),
                data=np.zeros(joined["itemID"].nunique()),
                name="FF",
            )
            item_list.loc[item_rel_freq.index] = item_rel_freq
            item_list = item_list.to_frame()
            item_list["group"] = g_value

            rel_freq[f"{group}_{g_value}"] = item_list
    return rel_freq


def calc_rel_freq_inter(joined, user_groups):
    rel_freq = {}
    for group, g_values in user_groups.items():
        for g_value in g_values:
            group_data = joined.filter_by([(group, "=", g_value)])
            item_rel_freq = item_relative_freq(group_data).to_frame()
            item_rel_freq["group"] = g_value
            rel_freq[f"{group}_{g_value}"] = item_rel_freq
    return rel_freq


def sample_most_pop(population, num_sample=5, prob_bins=[], group_key="group"):

    return


def calc_user_stereotyp_pref(ff_values, method="mean"):
    "This function scores a user to be stereotypical according to the scores of its items"
    user_stereo = -1.05
    if len(ff_values > 0):
        if method == "mean":
            user_stereo = np.mean(ff_values)  # [-1:1]
        elif method == "median":
            user_stereo = np.median(ff_values)  # [-1:1]
        elif method == "mean-abs":
            user_stereo = np.mean(np.abs(ff_values))  # [0:1]
        elif method == "median-abs":
            user_stereo = np.median(np.abs(ff_values))  # [0:1]
        elif method == "mean-pos":
            user_stereo = np.mean(np.where(ff_values >= 0))  # [0:1]
        elif method == "median-pos":
            user_stereo = np.median(np.where(ff_values >= 0))  # [0:1]
        elif method == "diff":
            # [0:1]
            user_stereo = (
                np.sum(np.where(ff_values >= 0))
                - np.sum(np.abs(np.where(ff_values < 0))) / 2
            )
        elif method == "inc_ratio":
            user_stereo = np.sum(np.where(ff_values >= 0)) / np.sum(
                np.abs(np.where(ff_values < 0))
            )
        else:
            raise Exception("Not implemented stereotypical user preferences measure")

    return user_stereo


def calculate_dataset_stereotyp_score(user_dataset, ff_data, sterotyp_method):
    unique_users = user_dataset["userID"].unique()
    user_ster = pd.Series(
        index=unique_users, data=np.zeros(len(unique_users)), name="user_ster"
    )
    user_ster.index.name = "userID"
    for user in unique_users:
        user_data = user_dataset.loc[user_dataset["userID"] == user]
        # Selecting only items that have defined FF values from the user profile
        valid_user_items = np.intersect1d(
            user_data["itemID"].values, ff_data.index.values
        )
        # print(len(valid_user_items),len(ff_data))
        user_ff_values = ff_data.loc[valid_user_items]
        # Estimating the stereotypicallity of the user profile
        user_stereo_pref = calc_user_stereotyp_pref(
            user_ff_values.values, method=sterotyp_method
        )
        user_ster.loc[user] = user_stereo_pref
    return user_ster


def calc_all_user_stereotyp_pref(user_dataset, ff_data, sterotyp_methods=STEREO_TYPES):
    scores = []
    for stereo_m in sterotyp_methods:
        ster_scores = calculate_dataset_stereotyp_score(user_dataset, ff_data, stereo_m)
        ster_scores.name = stereo_m
        scores.append(ster_scores)
    return pd.concat(scores, axis=1)


def calc_user_stereotyp_category_weights(ff_values):
    return np.sum(np.where(ff_values >= 0)), np.sum(np.abs(np.where(ff_values < 0)))


def calculate_dataset_pref_by_category(user_dataset, ff_data, sterotyp_method):
    unique_users = user_dataset["userID"].unique()

    data = []
    for user in unique_users:
        user_data = user_dataset.loc[user_dataset["userID"] == user]
        # Selecting only items that have defined FF values from the user profile
        user_ff_data = ff_data.loc[
            np.intersect1d(user_data["itemID"].unique(), ff_data.index.values)
        ]
        pos_ster, neg_ster = calc_user_stereotyp_category_weights(user_ff_data)
        data.append([user, pos_ster, neg_ster])

    user_ster = pd.DataFrame(
        data=np.zeros(len(unique_users), 3), columns=["userID", "pos_ster", "neg_ster"]
    )
    user_ster.set_index("userID", inplace=True)

    return user_ster

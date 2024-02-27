import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from pd_utils import filter_by
import matplotlib.pyplot as plt
from data_utils import * 
from stereo_utils import *
from constants import *


def generate_genre_inclination(data,out_dir,name):
    user_groups = {"gender":["M","F"]}
    rel_freq  = calc_rel_freq_inter(data,user_groups)
    ff = calculate_ff(rel_freq,"gender_M","gender_F")
    ff.to_csv(Path(out_dir)/f"{name}_gender_incl.csv")  
    
def preprocess_lfm():
    data_inter = pd.read_csv(ROOT_DIR/"lfm-multi-attr/Gustavo-2023-09-v2/inter.tsv.bz2",sep="\t",names=["userID","itemID","freq"],engine="python")
    data_user = pd.read_csv(ROOT_DIR/"lfm-multi-attr/Gustavo-2023-09-v2/demo.tsv.bz2",sep="\t",names=["userID","country","age","gender","created_at"],engine="python")
    joined = data_inter.merge(data_user[["userID","gender"]], on="userID")
    joined["gender"] = joined["gender"].str.upper()
    print(joined.nunique())
    joined=core_filtering(joined,K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/lfm-100k"
    if  not os.path.exists(f"{ROOT_DIR}/obfuscation/lfm-100k"):
        os.makedirs(out_dir)
    generate_genre_inclination(joined, out_dir,"lfm-100k")
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/lfm-100k/lfm-100k_inter.csv",index=False)
    print(joined.nunique())
    train_data, valid_data, test_data =  split_by_inter_ratio_recbole(joined)
    save_recbole_data(train_data,valid_data,test_data,out_dir)

def preprocess_ml1m():
    data_inter = pd.read_csv(ROOT_DIR/"ml-1m/ratings.dat",sep="::",names=["userID","itemID","rating","timestamp"],engine="python")
    data_user = pd.read_csv(ROOT_DIR/"ml-1m/users.dat",sep="::",names=["userID","gender","age","occcupation","zipcode"],engine="python")
    
    joined = data_inter.merge(data_user[["userID","gender"]], on="userID").dropna()
    print(joined.nunique())
    joined=core_filtering(joined,K_CORE)
    out_dir = f"{ROOT_DIR}/obfuscation/ml-1m"
    if  not os.path.exists(f"{ROOT_DIR}/obfuscation/ml-1m"):
        os.makedirs(out_dir)
    generate_genre_inclination(joined, out_dir,"ml-1m")
    print("Saving filtered dataset")
    joined.to_csv(f"{ROOT_DIR}/obfuscation/ml-1m/ml-1m_inter.csv",index=False)
    print(joined.nunique())
    train_data, valid_data, test_data =  split_by_inter_ratio_recbole(joined)
    save_recbole_data(train_data,valid_data,test_data,out_dir)

def core_filtering(data, min_k):
    while(True):
        item_user_counts = data.groupby(["itemID"])["userID"].nunique().reset_index()
        user_item_counts = data.groupby(["userID"])["itemID"].nunique().reset_index()
        
        valid_items=item_user_counts[item_user_counts['userID']>=min_k]['itemID'].values
        valid_users=user_item_counts[user_item_counts['itemID']>=min_k]['userID'].values
        
        result = data[data['itemID'].isin(valid_items) & data['userID'].isin(valid_users) ]
        
        item_user_counts = result.groupby(["itemID"])["userID"].nunique().reset_index()["userID"]
        user_item_counts = result.groupby(["userID"])["itemID"].nunique().reset_index()["itemID"]
        
        if ((item_user_counts>=min_k).any() and (user_item_counts>=min_k).any()):
            break
        else:
            data = result
    return result


if __name__ == "__main__":
    preprocess_lfm()
    preprocess_ml1m()
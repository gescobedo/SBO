#%%
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import sparse as sp

def split_by_inter_ratio(data, ratio=0.8, random_state=42,user_key="userID"
                        ):
    np.random.seed(random_state)
    data = data.sort_values([user_key],ascending=True)
    grouped = data.groupby(user_key).apply(lambda x : x.sample(n=int(ratio * len(x)),random_state=random_state).index.values)
    indexes_train = np.concatenate(grouped.values)
    data["tr"] = False
    data.loc[indexes_train,"tr"] = True
  
    train_data=data[data["tr"]] 
    test_data=data[~data["tr"]] 
    return train_data, test_data

def split_by_inter_ratio_recbole(data, ratio=0.8, random_state=42,user_key="userID"
):
    # train-test split
    train_data,test_data =split_by_inter_ratio(data,ratio,random_state,user_key)                       
    # train-valid split
    train_data,valid_data = split_by_inter_ratio(train_data,ratio,random_state,user_key)
    # correcting columns for recbole format
    train_data= transform_to_recbole(train_data)
    valid_data= transform_to_recbole(valid_data)
    test_data= transform_to_recbole(test_data)
    
    return train_data, valid_data, test_data 

def save_recbole_data (tr,val,te, out_dir):
    dataset_name = out_dir.split("/")[-1]
    tr= transform_to_recbole(tr)
    val= transform_to_recbole(val)
    te= transform_to_recbole(te)
    tr.to_csv(Path(out_dir)/f"{dataset_name}.train.inter",index=False)
    val.to_csv(Path(out_dir)/f"{dataset_name}.valid.inter",index=False)
    te.to_csv(Path(out_dir)/f"{dataset_name}.test.inter",index=False)



def read_dataset_to_obfuscate(data_dir):
    file_name = data_dir.split("/")[-1]
    train_data_url = f"{data_dir}/{file_name}.train.inter"
    valid_data_url = f"{data_dir}/{file_name}.valid.inter"
    test_data_url = f"{data_dir}/{file_name}.test.inter"
    dataset_name = file_name  
    incl_data_url = f"{data_dir}/{file_name}_gender_incl.csv"

    train_data=transform_to_obf(pd.read_csv(train_data_url))
    valid_data=transform_to_obf(pd.read_csv(valid_data_url))
    test_data=transform_to_obf(pd.read_csv(test_data_url))
    user_data =train_data.drop_duplicates(["userID","gender"]).reset_index()[["userID","gender"]]
    inclination_data = pd.read_csv(incl_data_url, index_col="itemID")
  
    return train_data, valid_data, test_data, inclination_data, user_data, dataset_name
#%%
def transform_to_recbole(data):
    recbole_map = {
        'userID':      'user_id:token',
        'itemID':     'item_id:token',
        'gender':    'gender:token',
        'timestamp':    'timestamp:float',
        'rating' : 'rating:token',
        'freq' : 'freq:float',
        'tr':    'tr:token',
        }
    data.rename(columns=recbole_map,inplace=True )
    return data

def transform_to_obf(data):
    recbole_map = {
        'user_id:token' : 'userID',
        'item_id:token' : 'itemID'     ,
        'gender:token' : 'gender',
        'timestamp:float' : 'timestamp',
        'rating:token' : 'rating',
        'freq:float' : 'freq',
        'tr:token': 'tr',

        }
    data.rename(columns=recbole_map,inplace=True )
    return data

def transform_dataframe_to_sparse(interactions):
    unique_items = interactions["itemID"].unique()
    print(f"n_items: {len(unique_items)}")
    item2token = pd.Series(unique_items)
    token2item = pd.Series(data=item2token.index, index=item2token.values)
    
    unique_users = interactions["userID"].unique()
    user2token = pd.Series(unique_users)
    token2user = pd.Series(data=user2token.index, index=user2token.values)
    
    
    #print(interactions.head())
    # assigning unique ids 
    interactions.loc[:,"userID"] = token2user.loc[interactions.loc[:,"userID"]].values
    interactions.loc[:,"itemID"] = token2item.loc[interactions.loc[:,"itemID"]].values
    
    user_info = interactions.drop_duplicates(["userID"])

    uids_iids_array = interactions[["userID","itemID"]].values
    n_users,n_items = interactions.userID.nunique(),len(unique_items) 
    data = np.ones(uids_iids_array.shape[0],dtype=np.int8)

    uids,iids = uids_iids_array[:,0],uids_iids_array[:,1]
    interaction_matrix = sp.csr_matrix((data, (uids, iids)), 
                                        (n_users, n_items))  
    return interaction_matrix,user_info,token2item,token2user
 
def save_csr_matrix(data_dir, interactions):

    unique_items = interactions["itemID"].unique()
    print(f"n_items: {len(unique_items)}")
    item2token = pd.Series(unique_items)
    token2item = pd.Series(data=item2token.index, index=item2token.values)
    
    unique_users = interactions["userID"].unique()
    user2token = pd.Series(unique_users)
    token2user = pd.Series(data=user2token.index, index=user2token.values)
    
    
    #print(interactions.head())
    # assigning unique ids 
    interactions.loc[:,"userID"] = token2user.loc[interactions.loc[:,"userID"]].values
    interactions.loc[:,"itemID"] = token2item.loc[interactions.loc[:,"itemID"]].values
    
    user_info = interactions.drop_duplicates(["userID"])

    uids_iids_array = interactions[["userID","itemID"]].values
    n_users,n_items = interactions.userID.nunique(),len(unique_items) 
    data = np.ones(uids_iids_array.shape[0],dtype=np.int8)

    uids,iids = uids_iids_array[:,0],uids_iids_array[:,1]
    interaction_matrix = sp.csr_matrix((data, (uids, iids)), 
                                        (n_users, n_items))
    sp.save_npz(os.path.join(data_dir, "interactions.npz"),interaction_matrix)
    user_info.to_csv(os.path.join(data_dir, "user_info.csv"),index=False)
    token2item.to_csv(os.path.join(data_dir,"item_mapping.csv"))
    token2user.to_csv(os.path.join(data_dir,"user_mapping.csv"))

   
    

# %%

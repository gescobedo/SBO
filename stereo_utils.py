import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
from pd_utils import filter_by
import matplotlib.pyplot as plt

def do_something_with_the_group_data(data):
    
    return ""
def item_relative_freq(data):
    # |U_i|/|U|
   
    rel_freq = (data.groupby("itemID")["userID"].nunique())/ data["userID"].nunique()
    rel_freq.name="FF"
    return rel_freq

def calculate_ff(rel_freq, x,y):
    
    common = np.intersect1d(rel_freq[x].index,rel_freq[y].index)
    common_concat = pd.concat([rel_freq[x].loc[common],rel_freq[y].loc[common]],axis=1)
    ff = (rel_freq[x]["FF"].loc[common] - rel_freq[y]["FF"].loc[common])/(1e-24+np.max(common_concat["FF"],axis=1))
    ff.name="FF"
    return ff

def calc_rel_freq(joined, user_groups):
    rel_freq  = {}
    for group, g_values in user_groups.items():
        for g_value in g_values:
            group_data = joined.filter_by([(group,"=",g_value)])
            item_rel_freq = item_relative_freq(group_data)
            item_list=pd.Series(index= joined["itemID"].unique(),data=np.zeros(joined["itemID"].nunique()),name="FF")
            item_list.loc[item_rel_freq.index]=item_rel_freq
            item_list =item_list.to_frame()
            item_list["group"] = g_value
            
            rel_freq[f"{group}_{g_value}"]  = item_list 
    return rel_freq
def calc_rel_freq_inter(joined, user_groups):
    rel_freq  = {}
    for group, g_values in user_groups.items():
        for g_value in g_values:
            group_data = joined.filter_by([(group,"=",g_value)])
            item_rel_freq = item_relative_freq(group_data).to_frame()            
            item_rel_freq["group"] = g_value
            rel_freq[f"{group}_{g_value}"]  = item_rel_freq
    return rel_freq
def sample_most_pop(population, num_sample=5,prob_bins=[], group_key="group"):
    
    return

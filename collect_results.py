import pandas as pd


results_files = ['random_completion-1000/test_BPR_2023-08-30 16:25:46.145796.pkl',
'random_completion-1000/test_BPR_2023-08-31 17:21:07.443649.pkl',
'random_completion-500/test_BPR_2023-08-31 00:48:08.670203.pkl',
'random_completion-500/test_BPR_2023-08-31 23:54:39.085208.pkl',
'true_negatives_shrink/test_BPR_2023-08-30 10:40:29.322245.pkl',
'true_negatives_shrink/test_BPR_2023-08-31 14:46:58.805373.pkl',
'random_completion-10000/test_BPR_2023-09-04 14:15:10.523891.pkl']

results_files = [
'random_completion-10000/test_BPR_2023-09-04 14:15:10.523891.pkl',
'random_completion-10000/test_LightGCN_2023-09-07 06:22:01.612544.pkl',
'random_completion-10000/test_NeuMF_2023-09-06 16:56:48.313282.pkl',
'random_completion-10000/test_LightGCN_2023-09-08 17:29:35.080365.pkl',
'random_completion-10000/test_LightGCN_2023-09-09 17:37:10.221435.pkl',
]
results_files= [

'right-10000/test_BPR_2023-09-13 12:30:08.207331.pkl', 
'right-10000/test_LightGCN_2023-09-14 02:28:45.780509.pkl',
'right-10000/test_NeuMF_2023-09-13 16:55:05.256836.pkl',
'right-10000/test_BPR_2023-09-14 12:04:32.508211.pkl', 
'right-10000/test_LightGCN_2023-09-14 19:54:03.404005.pkl',
'right-10000/test_NeuMF_2023-09-14 13:58:58.566903.pkl'
]
#%%

import pickle
def convert_table(file):
    for res in file:
        res.update(res["test_result"])
   
    df = pd.DataFrame.from_dict(file)
    #print(df)
    converted_df=df.groupby(["Model","dataset"])[[x for x in df.columns if  x.endswith("10")]].mean()
    std_df=df.groupby(["Model","dataset"])[[x for x in df.columns if  x.endswith("10")]].std()
    return df,converted_df,std_df

from collections import defaultdict
results_dict = defaultdict(lambda:[])
for file_name in results_files:
    data=pickle.load(open(file_name,"rb"))
    key = file_name.split("/")[0]
    for data_item in data:
        results_dict[key].append(data_item)
processed_results_dict = {}
dfs  = []
for key,data in results_dict.items():
    df, conv_df, std = convert_table(data)
    df["key"] =key
    processed_results_dict[key]=conv_df
    print(key)
    print(processed_results_dict[key])
    print(std)
    dfs.append(df)
pd.concat(dfs,axis=0,ignore_index=True).to_csv("result_table-right.csv",index=False)



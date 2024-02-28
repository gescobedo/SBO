import numpy as np
from multiprocessing import Pool
import pandas as pd
from datetime import datetime
import pickle
import os
import argparse
from tqdm import tqdm
from recbole.quick_start import run_recbole, run_recboles# run_recbole_conf
import torch
from joblib import delayed, Parallel
from recbole.utils import list_to_latex
from pathlib import Path

config_base = {}
datasets  = [
        '',
        #'_remove_0.05_ff_median_th0.005', 
        #'_remove_0.1_ff_median_th0.005', 
        #'_remove_0.15_ff_median_th0.005',
        '_remove_0.05_ff_mean_th0.005', 
        '_remove_0.1_ff_mean_th0.005', 
        #'_remove_0.15_ff_mean_th0.005', 
]
        
        

models = ["BPR",
          "LightGCN",
          "MultiVAE"
          ]
num_cores= 6

parameter_dict = {
    "field_separator": "," ,
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "load_col":   {"inter": ["user_id","item_id"]},
    "save_dataset": True,
    "save_dataloaders": True,
    "embedding_size": 64,
    "epochs": 100, 
    "train_batch_size": 512,
    "eval_batch_size": 2048, 
    "benchmark_filename": ['train','valid','test'] ,
    "stopping_step": 10,
    "metrics": ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision','ItemCoverage'],
    "topk": [10,20,50],
    "metric_decimal_place":6,
    "valid_metric": "NDCG@10",
    #'train_neg_sample_args': {
    #    'distribution':"user_custom",
    #    'user_pool': "",
    #    'mode_sampler': "true_negatives_shrink"
    #},
    "use_gpu":True,
    #"device": torch.device('cpu')
    #"nproc":8,
    #"gpu_id": '1,2,3',     
}
def run_alg(args, model, dataset, config):
    #model, dataset, config =args


    data_path = config["out_dir"]
    config["checkpoint_dir"] = data_path + "saved/"+dataset+"/"
    config["dataset_save_path"]= data_path + "saved/"+dataset+f"/{model}_dataset.pth"
    config["dataloaders_save_path"]= data_path + "saved/"+dataset+f"/{model}_dataloader.pth"

    config["world_size"]= args.world_size
    config["ip"]= args.ip
    config["port"]= args.port
    config["nproc"]= args.nproc
    config["offset"]= args.group_offset

    if  not os.path.exists(config["checkpoint_dir"]):
        os.makedirs(config["checkpoint_dir"])
    params  = {
                    "model":model,
                    "dataset":dataset,
                    "config_file_list":None,
                    "config_dict":config
                }
    #results = mp.spawn(
    #        run_recbole_conf,
    #        args=(model, dataset,None,config           
    #        ),
    #        nprocs=args.nproc,
    #        join=False
    #        
    #)
   
    results = run_recbole(model=model,dataset=dataset,config_dict=config)
    results['name'] = f"{model}-{dataset}"
    
    return results
    

# Constructing parameter pool
  
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='opt_clustering')
    
    parser.add_argument('--model', required=True,help='path to Test file')
    parser.add_argument('--data_path', required=True,help='path to Test file')
    parser.add_argument('--out_dir',required=False)
    parser.add_argument('--gpu',type=int,required=False)

    parser.add_argument("--valid_latex", type=str, default="./latex/valid.tex", help="config files"
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

    model = args.model
    models = args.model.split(",")
    parameter_dict["data_path"] = args.data_path
    parameter_dict["out_dir"] = args.out_dir
    if  not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    params = []
    gpu_ids = [1,2,3]*20
    gpu_id = 0
    

    for model in models:
        for dataset in datasets:
            
            config = parameter_dict.copy()
            config["gpu_id"] = args.gpu
            #config["device"] = torch.device(f'cuda:{gpu_ids[gpu_id]}')
            gpu_id+=1    
            params.append((args,model,dataset,config))
    
    from recbole.quick_start import run_recboles
    import torch.multiprocessing as mp
    results_months = {}
    valid_result_list = []
    test_result_list = []
    #for dataset in datasets:
    results_list = []
    #print(params)    
    results_list = Parallel(n_jobs=args.nproc)(delayed(run_alg)(*param) for param in params)
    for res_dict in results_list:
        valid_res_dict = {"Model": model}
        test_res_dict = {"Model": model}
        result = res_dict.copy()
        valid_res_dict["best_valid_result"]=result["best_valid_result"]
        valid_res_dict["name"]=result["name"]
        test_res_dict["test_result"]=result["test_result"]
        test_res_dict["name"]=result["name"]
        
        #valid_res_dict["user_pool"]=result["user_pool"]
        #test_res_dict["user_pool"]=result["user_pool"]
        
        bigger_flag = result["valid_score_bigger"]
        subset_columns = list(result["best_valid_result"].keys())
        valid_result_list.append(valid_res_dict)
        test_result_list.append(test_res_dict)

    #valid_result_list = []
    #test_result_list = []
    #run_times = len(months)*len(user_pools)

    
    #for idx in range(run_times):
    #    model, dataset, config = params[idx]

    #    valid_res_dict = {"Model": model}
    #    test_res_dict = {"Model": model}
    #    result = run_alg(args, model, dataset, config)
    #    valid_res_dict.update(result["best_valid_result"])
    #    valid_res_dict.update(result["name"])
    #    test_res_dict.update(result["test_result"])
    #    test_res_dict.update(result["name"])
    #    bigger_flag = result["valid_score_bigger"]
    #    subset_columns = list(result["best_valid_result"].keys())
#
    #    valid_result_list.append(valid_res_dict)
    #    test_result_list.append(test_res_dict)
    print(f"Saving results to {args.out_dir}")

    pickle.dump(valid_result_list,open(Path(args.out_dir)// f"valid_{args.model}_{str(datetime.now())}.pkl","wb"))    
    pickle.dump(test_result_list,open(Path(args.out_dir)//f"test_{args.model}_{str(datetime.now())}.pkl","wb"))    
    
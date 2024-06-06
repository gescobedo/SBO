import os
import shutil
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed

import torch
from torch.optim import Adam

import global_config
import src.modules.parallel
from src.config_classes.atk_config import AtkConfig
from src.data.user_feature import FeatureDefinition, FeatureType
from src.logging.logger import Logger
from src.modules.evaluation import AdvEval
from src.modules.losses import AdvLosses
from src.modules.polylinear import PolyLinearParallel
from src.logging.utils import redirect_to_tqdm
from src.utils.utils import postprocess_loss_results, add_to_dict, scale_dict_items,preprocess_data
from src.utils.input_validation import parse_input
from src.data.data_loading import get_datasets_and_loaders
from src.utils.helper import reproducible, yaml_load, \
    get_device_matching_current_process, json_dump, pickle_dump, create_unique_names, json_load


def attack(data_path, dataset, fold, results_dir, attacker_config,
           n_workers, devices, is_verbose=False):
    device = get_device_matching_current_process(devices)
    loading_fn, preprocess_fn,  = get_datasets_and_loaders, preprocess_data

    if is_verbose:
        print("Attacking", dataset)

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(results_dir)

        # gather configuration for attackers
        atk_configs = attacker_config.get("atk_groups", [])
        atk_configs = [AtkConfig(**config) for config in atk_configs]
        for cfg, name in zip(atk_configs, create_unique_names([cfg.feature for cfg in atk_configs])):
            cfg.group_name = name
        
        #print("....")
        #print(attacker_config["fixed_params"]["optim"])
        # load data
        features = [FeatureDefinition(name=cfg.feature, type=FeatureType.from_str(cfg.type)) for cfg in atk_configs]
        dataset_and_loaders = loading_fn(data_path=data_path,dataset_name=dataset, fold=fold, features=features,
                                         splits=("train", "val", "test"), run_parallel=True, n_workers=n_workers,
                                                                                  )
        tr_set, tr_loader = dataset_and_loaders["train"]
        vd_set, vd_loader = dataset_and_loaders["val"]
        te_set, te_loader = dataset_and_loaders["test"]

      

        attacker, optim, loss_fn, eval_fn = setup_attacker(input_size=tr_set.n_items   ,
                                                           full_config=attacker_config, atk_configs=atk_configs,
                                                           device=device)

        for epoch in trange(attacker_config["epochs"], desc="Epochs", position=1, leave=True, disable=not is_verbose):
            # train attacker
            attacker.train()
            run_epoch("train", attacker, preprocess_fn, loss_fn, eval_fn, tr_loader,
                      device, epoch, logger, optim=optim, return_raw_results=False)

            # validate attacker
            attacker.eval()
            with torch.no_grad():
                run_epoch("val", attacker, preprocess_fn, loss_fn, eval_fn, vd_loader,
                          device, epoch, logger, optim=None, return_raw_results=False)

        # test attacker
        attacker.eval()
        with torch.no_grad():
            eval_dict, raw_results = run_epoch("test", attacker, preprocess_fn, loss_fn, eval_fn,
                                               te_loader, device, epoch, logger, optim=None, return_raw_results=True)

        # Store attacker data to be able to analyze its performance later on
        pickle_dump({
            **raw_results,
            "user_feature_map": {k: v.value_map for k, v in te_set.user_features.items() if v.is_categorical_feature}
        }, os.path.join(results_dir, f"test_set_attacker_data.pkl"))

        # some weird yaml error prevents saving the dict, therefore use json for now
        json_dump(eval_dict, os.path.join(results_dir, f"test_set_attacker_evaluation.json"))
        eval_dict["dataset"]=dataset
        eval_dict["fold"]=fold
        return eval_dict

def run_epoch(split, attacker, preprocess_fn, loss_fn, eval_fn, data_loader, device, epoch, logger,
              optim=None, return_raw_results=False):
    aggregated_loss_dict = defaultdict(lambda: 0)
    aggregated_eval_dict = defaultdict(lambda: 0)
    sample_count, aggregated_loss = 0, 0.

    raw_data = defaultdict(lambda: list())

    for indices, *model_input, _, targets in tqdm(data_loader, desc="Steps", position=2, leave=True, disable=True):
        n_samples = len(indices)
        sample_count += n_samples

        targets = [t.to(device) for t in targets]
        model_input = preprocess_fn(model_input, device)

        logits = attacker(*model_input)
        result = loss_fn(logits, targets)

        loss, loss_dict = postprocess_loss_results(result)
        eval_dict = eval_fn(logits, targets)

        if optim is not None:
            # Update model
            optim.zero_grad()
            loss.backward()
            optim.step()

        # store results over batches
        aggregated_loss += loss * n_samples
        aggregated_loss_dict = add_to_dict(aggregated_loss_dict, loss_dict, multiplier=n_samples)
        aggregated_eval_dict = add_to_dict(aggregated_eval_dict, eval_dict, multiplier=n_samples)

        if return_raw_results:
            raw_data["indices"].append(indices)
            raw_data["logits"].append([[log.detach().cpu() for log in log_grp] for log_grp in logits])
            raw_data["targets"].append([tar.detach().cpu() for tar in targets])

    logger.log_value(f"{split}/atk_loss", aggregated_loss / sample_count, epoch)
    logger.log_value_dict(f"{split}/atk_loss", scale_dict_items(aggregated_loss_dict, 1 / sample_count), epoch)

    aggregated_eval_dict = scale_dict_items(aggregated_eval_dict, 1 / sample_count)
    logger.log_value_dict(f"{split}/atk_eval", aggregated_eval_dict, epoch)

    if return_raw_results:
        raw_data = dict(raw_data)
        raw_data["indices"] = np.concatenate(raw_data["indices"])
        raw_data["logits"] = [[np.concatenate(log) for log in zip(*grp)] for grp in zip(*raw_data["logits"])]
        raw_data["targets"] = [np.concatenate(grp) for grp in zip(*raw_data["targets"])]
        return aggregated_eval_dict, raw_data


def setup_attacker(input_size: int, full_config: dict, atk_configs: List[AtkConfig], device: torch.device):
    attacker_modules = [
        PolyLinearParallel(layer_config=[input_size] + config.dims,
                           n_parallel=config.n_parallel,
                           input_dropout=config.input_dropout,
                           activation_fn=config.activation_fn
                           )
        for config in atk_configs
    ]

    attacker = src.modules.parallel.Parallel(modules=attacker_modules,
                                             parallel_mode=src.modules.parallel.ParallelMode.SingleInMultiOut)
    attacker.to(device)
    #print(attacker_modules)
    loss_fn = AdvLosses(atk_configs)
    eval_fn = AdvEval(atk_configs)
    optim_params = full_config["optim"].copy()
    optim = Adam(attacker.parameters(), **optim_params)

    return attacker, optim, loss_fn, eval_fn


if __name__ == "__main__":

    input_config = parse_input("attacker", options=["atk_config", "data_path","datasets_file", "gpus","atk_results_dir",
                                                    "n_folds", "model_pattern", "n_parallel", "n_workers"],
                               access_as_properties=False)

    out_dir = input_config["atk_results_dir"]
    data_path = input_config["data_path"]
    # atk_results_dir must contain a JSON file
    datasets_file = json_load(os.path.join(out_dir,input_config["datasets_file"]))["datasets"]
    
    params = []
    for dataset in datasets_file:
        for fold in range(input_config["n_folds"]):
            
            attacker_config = yaml_load(input_config["atk_config"])
            attacker_name = input_config["dataset"]
            results_dir  = os.path.join(out_dir,attacker_name,dataset,"atk", str(fold))
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir, ignore_errors=True)
            params.append({
                "data_path":data_path,
                "dataset_name": dataset,
                "fold": fold,
                "results_dir": results_dir,
                "attacker_config": attacker_config["fixed_params"],
            })
            #print(attacker_config)

    devices = input_config["devices"]
    n_devices = len(devices)
    is_verbose = len(params) == 1 or (n_devices == 1 and input_config["n_parallel"] == 1)

    arguments = {
        
        "n_workers": input_config["n_workers"],
        "devices": input_config["devices"],
        "is_verbose": is_verbose
    }

    print("Running", len(params), "training jobs")
    results_folds = Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
        delayed(attack)(**arguments, **p) for p in tqdm(params, desc="Attacking experiments", position=0, leave=True))
    print(results_folds)
    import pandas as pd
    df_result=pd.DataFrame(results_folds)
    print(df_result)
    df_result.to_csv(os.path.join(out_dir,input_config["dataset"], f"test_set_attacker_evaluation.csv"), index=False)
    
    grouped= df_result.groupby("dataset")["gender_bacc"].agg({"mean","std"}).reset_index()

    grouped.to_csv(os.path.join(out_dir,input_config["dataset"], f"test_set_attacker_evaluation_grouped.csv"), index=False)
    
    # 
from global_config import MAX_FOLDS


input_options = {
    "experiment_type": {
        "type": str, "required": True, "choices": ['standard', 'up_sample']
    },

    "data_path": {
        "type": str, "required": False, "default": "../Datasets", 
        "help": "The dataset root directory."
    },
    "datasets_file": {
        "type": str, "required": False, "default": None, 
        "help": "JSON file inside 'atk_results_dir' containing a list datasets names to be attacked  {'datasets': ['ml-1m','lfm']}"
    },
      
    "atk_results_dir": {
        "type": str, "required": True, 
        "help": "Results folder for attacker networks"
    },

    "gpus": {
        "type": str, "required": False, "default": "",
        "help": "The gpus to run the models on, use e.g., '0,2' to run on GPU '0' and '2'"
    },

    "n_folds": {
        "type": int, "required": False, "default": MAX_FOLDS, "choices": range(1, MAX_FOLDS + 1),
        "help": "The number of folds to run on."
    },

    "n_workers": {"type": int, "required": False, "default": 0,
                  "help": "The number of workers that each dataloader should use"},

    "n_parallel": {
        "type": int, "required": False, "default": 1,
        "help": "The number of processes that should be run on each device"
    },

    "store_best": {
        "type": bool, "required": False, "default": False,
        "help": "Whether the best models found for each run should be stored, "
                "i.e., whether early stopping should be performed."
    },

    "store_every": {
        "type": int, "required": False, "default": 0, "choices": range(0, 100),
        "help": "After which number of epochs the model should be stored, 0 to deactivate this feature"
    },

    "config": {
        "type": str, "required": True,
        "help": "The config file to use for running an experiment"
    },

    "atk_config": {
        "type": str, "required": True,
        "help": "The attacker config file to use for running an experiment (only in case of executing train+atk)"
    },

    "split": {
        "type": str, "required": False, "default": "test", "choices": ["train", "val", "test"],
        "help": "The split to use."
    },

    "use_tensorboard": {
        "type": bool, "required": False, "default": False,
        "help": "Whether additional information should be logged via tensorboard"
    },

    "experiment": {"type": str, "required": False, "default": None,
                   "help": "The path to an experiment, i.e., collection of multiple runs, "
                           "where each one should be validated"},

    "run": {"type": str, "required": False, "default": None,
            "help": "The path to a run that should be validated."},

    "model_pattern": {
        "type": str, "required": False, "default": "*.pt*",
        "help": "If specified, only models that match this pattern are considered. "
                "(glob syntax is used)"
    },

    "oversample_ratio": {
        "type": float, "required": False, "default": 1,
        "help": "Sets the ratio of balanced-ness for the groups in the dataset. "
                "1 for balanced, 0 to deactivate oversampling."
    },

    "perform_undersampling": {
        "type": bool, "required": False, "default": False,
        "help": "Whether to perform undersampling of majority group than oversampling minority group. "
                "This option leads to ignore the option 'oversample_ratio' (not available in imblearn)."
    },

    "group_ratio": {
        "type": int, "required": False, "default": -1,
        "help": "Whether to perform under1sampling of majority group than oversampling minority group. "
                "This option leads to ignore the option 'oversample_ratio' (not available in imblearn)."
    },

    "features": {
        "type": str, "required": False, "default": None, "nargs": "+",
        "help": "The feature(s) to use for adversarial training. Note that they must align with the configuration files"
    }
}

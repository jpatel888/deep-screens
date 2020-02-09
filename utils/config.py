import json
from bunch import bunchify
import os
from utils.utils import get_args
from utils.utils import get_dict_from_json


def get_config_from_json(config_file_path):
    config_dict = get_dict_from_json(config_file_path)
    config = bunchify(config_dict)
    return config, config_dict


def process_config(json_file_name):
    config, _ = get_config_from_json(json_file_name)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
    config.figure_dir = os.path.join("../experiments", config.exp_name, "generated_figures/")
    return config


def get_default_config():
    try:
        return process_config("./configs/default.json")
    except json.decoder.JSONDecodeError:
        print("Error in JSON")
        exit(0)
    except:
        print("error in config")
        exit(0)


def get_config():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        config = get_default_config()
    return config

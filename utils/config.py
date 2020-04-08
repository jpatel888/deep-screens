import json
from bunch import bunchify
import os
from utils.utils import get_args
from utils.utils import get_dict_from_json


def get_config_from_json(config_file_path):
    config_dict = get_dict_from_json("values/" + config_file_path)
    config = bunchify(config_dict)
    return config, config_dict


def process_configs(json_file_names):
    configs = []
    for json_file_name in json_file_names:
        config, _ = get_config_from_json(json_file_name)
        config.summary_dir = os.path.join("./experiments", "summary/", config.exp_name + "/")
        config.checkpoint_dir = os.path.join("./experiments", "checkpoint/", config.exp_name + "/")
        config.figure_dir = os.path.join("./experiments", "generated_figures/", config.exp_name + "/")
        config.tflite_dir = os.path.join("./experiments", "tflite/", config.exp_name + "/")
        configs.append(config)
    return configs


def get_all_available_configs():
    all_config_paths = [path for path in os.listdir("values") if path.endswith("config.json")]
    all_config_paths.sort()
    return all_config_paths


def get_default_configs():
    print("Couldn't get config params or none provided, running all configs in configs/ sequentially")
    all_config_paths = get_all_available_configs()
    try:
        return process_configs(all_config_paths)
    except json.decoder.JSONDecodeError:
        print("Error in JSON")
        exit(0)
    except Exception as exception:
        print("Error in fetching config:", exception)
        exit(0)


def get_configs():
    try:
        args = get_args()
        configs = process_configs(args.config.split(","))
    except:
        configs = get_default_configs()
    print("Successfully loaded", len(configs), "configs")
    return configs

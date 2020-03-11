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
        config.summary_dir = os.path.join("./experiments", config.exp_name, "summary/")
        config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoint/")
        config.figure_dir = os.path.join("./experiments", config.exp_name, "generated_figures/")
        config.tflite_dir = os.path.join("./experiments", config.exp_name, "tflite/")
        configs.append(config)
    return configs


def get_default_configs():
    print("Couldn't get config params or none provided, using default")
    try:
        return process_configs(["default_config.json"])
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
        print("Successfully loaded", len(configs), "configs")
    except:
        configs = get_default_configs()
    return configs

import argparse
import sys
import json
import os


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def add_sys_paths(paths):
    for path in paths:
        sys.path.append(path)


def get_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict

import os
import imageio
import numpy as np
from data_loader.data_utils.data_augmenter import DataAugmenter
from collections import Counter
from utils.utils import get_dict_from_json


class DataUtils:
    def __init__(self, config, model):
        self.config = config
        self.model_output_height = model.get_output_height
        self.model_output_width = model.get_output_width
        self.augmenter = DataAugmenter(config)
        self.model_output_depth = (self.config.num_classes + 5) * len(self.config.anchor_boxes)
        self.root_data_dirs = {"train": self.config.root_train_dir, "test": self.config.root_test_dir}
        self.num_defect_categories = self.config.num_defect_categories
        self.anchor_boxes = self.config.anchor_boxes
        self.valid_train_dates = {}
        self.get_valid_dates()

    def get_valid_dates(self):
        """
        Sets a class variable to a list of valid date strings
            A valid date string is associated with three files
        :return: NA
        """
        for data_pool, root_data_dir in self.root_data_dirs.items():
            all_train_timestamps = [file_name.split("_") for file_name in os.listdir(root_data_dir)]
            train_timestamps_count = dict(Counter(all_train_timestamps))
            filtered_input_timestamps = {k: v for k, v in train_timestamps_count.items() if v == 3}
            self.valid_train_dates[data_pool] = list(filtered_input_timestamps.keys())

    def get_input(self):
        """
        Uses valid dates to compile an object nd_array of paths to images
        :return: object type nd_array of strings
        """
        inputs = {}
        for data_pool, root_data_dir in self.root_data_dirs.items():
            inputs[data_pool] = np.array(
                [(root_data_dir + valid_train_date + "_(baseline)" + self.config.image_extension,
                  root_data_dir + valid_train_date + "_(current)" + self.config.image_extension)
                 for valid_train_date in self.valid_train_dates[data_pool]]
            , dtype=object)
        return inputs

    def get_y(self):
        """
        Uses valid dates to compile an object nd_array of paths to images
        :return: object type nd_array of strings
        """
        ys = {}
        for data_pool, root_data_dir in self.root_data_dirs.items():
            ys[data_pool] = np.array(
                [root_data_dir + valid_train_date + "_(label)" + self.config.image_extension
                 for valid_train_date in self.valid_train_dates[data_pool]]
                , dtype=object)
        return ys

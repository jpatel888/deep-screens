import os
import imageio
import numpy as np
from data_loader.data_utils.data_augmenter import DataAugmenter
from data_loader.data_utils.defect import Defects
from collections import Counter
from utils.utils import get_dict_from_json
from bunch import bunchify


class DataUtils:
    def __init__(self, config):
        self.config = config
        self.model_output_height, self.model_output_width, self.model_output_depth = tuple(self.config.model_output_size)
        self.augmenter = DataAugmenter(config)
        self.root_data_dirs = {"train": self.config.root_train_dir, "test": self.config.root_test_dir}
        self.num_defect_categories = len(self.config.defect_types)
        self.valid_train_dates = {}
        self.set_valid_dates()

    def set_valid_dates(self):
        """
        Sets a class variable to a list of valid date strings
            A valid date string is associated with three files
        :return: NA
        """
        for data_pool, root_data_dir in self.root_data_dirs.items():
            all_train_timestamps = [file_name.split("_")[0] + "_" + file_name.split("_")[1] for file_name in os.listdir(root_data_dir)]
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
                [root_data_dir + valid_train_date + "_(labels)" + self.config.label_extension
                 for valid_train_date in self.valid_train_dates[data_pool]]
                , dtype=object)
        return ys

    def prepare_batch(self, input_paths, label_paths):
        batch_images, labels = [], []
        for input_path_pair, label_path in zip(input_paths, label_paths):
            input_image = DataUtils.input_paths_to_image_input(input_path_pair)
            label_json_bunch = self.label_json_path_to_label(label_path)
            batch_images.append(input_image)
            labels.append(self.labels_json_to_grid(label_json_bunch))
        return np.array(batch_images), np.array(labels)

    def get_grid_xy_indexes(self, label_grid, defect):
        _loc_x = defect.location[0]
        label_grid_cell_width = self.config.input_shape[0] / label_grid.shape[0]
        defect_label_grid_x_index = int(_loc_x / label_grid_cell_width)
        _loc_y = defect.location[1]
        label_grid_cell_height = self.config.input_shape[1] / label_grid.shape[1]
        defect_label_grid_y_index = int(_loc_y / label_grid_cell_height)
        return defect_label_grid_x_index, defect_label_grid_y_index

    def get_iou(self, defect, anchor):
        _ = self
        defect_width, defect_height = defect.location[2], defect.location[3]
        anchor_width, anchor_height = anchor.box_width, anchor.box_height
        x_intersection_length = max(defect_width, anchor_width) - (abs(anchor_width - defect_width) / 2)
        y_intersection_length = max(defect_height, anchor_height) - (abs(anchor_height - defect_height) / 2)
        intersection = x_intersection_length * y_intersection_length
        union = ((defect_width * defect_height) + (anchor_width * anchor_height)) - intersection
        return intersection/union

    def get_anchor_box_index(self, defect):
        """

        :param defect: defect object in label json
        :return:
        """
        max_iou_idx, max_iou = max(
            enumerate(self.config.anchor_boxes),
            key=lambda idx, anchor: self.get_iou(defect, anchor)
        )
        return max_iou_idx

    def labels_json_to_grid(self, label_json_bunch):
        """
        Converts the label json into a feed_dict ready nd_array
        :param label: use conf. input size to reshape image and labels, augment data
        :return: list of 4D np.uint8 array of feature maps
        """
        defects = label_json_bunch.defects
        img_height = label_json_bunch.img_height
        img_width = label_json_bunch.img_width
        blank_grid = np.zeros((self.model_output_height, self.model_output_width, self.model_output_depth))
        return Defects(self.config, defects, img_height, img_width).generate_grid(blank_grid)

    @staticmethod
    def input_paths_to_image_input(input_path_pair):
        """
        reads images from paths
        :param input_path_pair: tuple of input path strings
        :return: return nd_array of images concatenated along last axis
        """
        baseline_path, current_path = input_path_pair
        return np.concatenate((imageio.imread(baseline_path),
                               imageio.imread(current_path)), axis=-1)

    @staticmethod
    def label_json_path_to_label(label_path):
        return bunchify(get_dict_from_json(label_path))

import os
import imageio
import numpy as np
from data_loader.data_utils.data_augmenter import DataAugmenter
from collections import Counter
from utils.utils import get_dict_from_json
from bunch import bunchify


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

    def prepare_batch(self, input_paths, label_paths):
        batch_images, labels = [], []
        for input_path_pair, label_path in zip(input_paths, label_paths):
            input_image = DataUtils.input_paths_to_image_input(input_path_pair)
            label = self.label_json_path_to_label(label_path)
            input_name, label = self.augmenter(image=input_image, label=label)
            batch_images.append(input_name)
            labels.append(self.labels_json_to_grid(label))
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

    def add_defect(self, label_grid, defect):
        """
        places defect in label_grid and returns
        :param label_grid: grid for feature map labels
        :param defect: defect object in label json
        :return:
        """
        grid_cell_width = self.config.input_shape[0] / label_grid.shape[0]
        grid_cell_height = self.config.input_shape[1] / label_grid.shape[1]
        defect.defect_id = self.config.defect.categories.index(defect.defect_type)
        x_index, y_index = self.get_grid_xy_indexes(label_grid, defect)
        anchor_box_index = self.get_anchor_box_index(defect)
        x_delta = (grid_cell_width * (x_index + 0.5)) - defect.location[0]
        y_delta = (grid_cell_height * (y_index + 0.5)) - defect.location[1]
        width_delta = defect.location[2] / self.config.anchor_boxes[anchor_box_index].box_width
        height_delta = defect.location[3] / self.config.anchor_boxse[anchor_box_index].box_height
        confidence_idx, confidence = 0, 1
        category_idx, category_value = defect.defect_id, 1
        xywh = np.array([x_delta, y_delta, width_delta, height_delta])
        base_index = (self.config.num_classes + 5) * anchor_box_index
        label_grid[x_index, y_index, base_index + confidence_idx] = confidence
        label_grid[x_index, y_index, base_index + category_idx] = category_value
        label_grid[x_index, y_index, base_index + self.config.num_classes + 1: base_index + self.config.num_classes + 5] = xywh
        return label_grid

    def labels_json_to_grid(self, label):
        """
        TODO: Needs push multiple outputs to backprop every 2 layers
        Converts the label json into a feed_dict ready nd_array
        :param label: use conf. input size to reshape image and labels, augment data
        :return: list of 4D np.uint8 array of feature maps
        """
        defects = label.defects
        label_grid = np.zeros((self.model_output_width, self.model_output_height, self.model_output_depth))
        for defect in defects:
            label_grid = self.add_defect(label_grid, defect)
        return label_grid

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

import numpy as np
import cv2


class DataAugmenter:
    def __init__(self, config):
        self.config = config
        self.image_input_shape = self.config.input_shape

    def __call__(self, *args, **kwargs):
        """
        Add any other inputs to **kwargs and methods for augmentation to the class below,
            and call them in this method
        Note: This is an inefficient pipeline, since the data is simply arrays of path,
            However, this is good for low memory machines
        :param kwargs: kwargs: At the minimum should include input_image and labels
        :return:
        """
        image = kwargs["input_image"]
        label_json = kwargs["label"]
        image, label_json = self.scale(image, label_json)
        image, label_json = self.shift(image, label_json)
        return image, label_json

    def shift(self, image, label, shift=None):
        """
        Places a scaled image randomly in a grad of zeros in the shape
            of the input tensor defined in self.config
        :param image: input image len(shape) == 3
        :param label: bunchified json object
        :param shift: if None, a random shift will be applied in the window
        :return:
        """
        _input = np.zeros(self.image_input_shape + [image.shape[-1]])
        tensor_input_height = self.image_input_shape[0]
        tensor_input_width = self.image_input_shape[1]
        image_height = image.shape[0]
        image_width = image.shape[1]
        vertical_shift, horizontal_shift = shift if shift is not None else (None, None)
        if shift is None:
            max_vertical_shift = tensor_input_height - image_height
            max_horizontal_shift = tensor_input_width - image_width
            vertical_shift = np.random.randing(0, high=max_vertical_shift)
            horizontal_shift = np.random.randint(0, high=max_horizontal_shift)
        _input[vertical_shift:vertical_shift + image_height, horizontal_shift:horizontal_shift + image_width] = image
        for defect_idx in range(len(label.defects)):
            label.defects[defect_idx].location[0] += horizontal_shift
            label.defects[defect_idx].location[1] += vertical_shift
        return _input, label

    def scale(self, image, label, scalar=None):
        """
        Scale the image and modify the labels by scalar
        :param image: input image len(shape) == 3
        :param label: bunchified json object
        :param scalar: if None, a random scalar will be applied
        :return:
        """
        if scalar is None:
            image_longer_index = 0 if image.shape[0] > image.shape[1] else 1 # replace with max(lambda...)
            image_longer_length = image.shape[image_longer_index]
            min_scalar = (self.config.min_scalar *
                          self.image_input_shape[image_longer_index]) / image_longer_length
            max_scalar = self.image_input_shape[image_longer_index] / image_longer_length
            scalar = float(np.random.uniform(low=min_scalar, high = max_scalar, size=()))
            print(scalar)
        new_image_hw = tuple((np.array(image.shape)))
        new_image = cv2.resize(image, dsize=(new_image_hw[1], new_image_hw[0]), interpolation=cv2.INTER_CUBIC)
        for defect_idx in range(len(label.defects)):
            for box_idx in range(4):
                label.defects[defect_idx].location[box_idx] *= scalar
        return new_image, label

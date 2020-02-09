import numpy as np


def split_images_by_depth(concatenated_images, image_depth=3):
    assert concatenated_images.shape[2] % image_depth == 0
    split_images = tuple(np.split(concatenated_images, concatenated_images.shape[2] / image_depth, axis=-1))
    return split_images


def stack_images_by_depth(list_of_images):
    return np.concatenate(list_of_images, axis=-1)


def concatenate_images_by_width(list_of_images):
    return np.concatenate(list_of_images, axis=1)


def concatenate_images_by_height(list_of_images):
    return np.concatenate(list_of_images, axis=0)


from data_loader.data_utils.utils import DataUtils
import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.data_utils = DataUtils(config)
        self.input = self.data_utils.get_input()
        self.y = self.data_utils.get_y()
        print(self.input, self.y)

    def roll_data_for_batch(self, batch_size, data_pool):
        """
        Rolls data after use for new batch at next "next_batch" call
        :param batch_size: number of images per iteration, defined in config
        :param data_pool:
        :return:
        """
        self.input[data_pool] = np.roll(self.input[data_pool], batch_size, axis=0)
        self.y[data_pool] = np.roll(self.y[data_pool], batch_size, axis=0)

    def next_batch(self, batch_size, data_pool='train'):
        """
        pulls and prepares data for iteration run
        :param batch_size: number of images per iteration, defined in config
        :param data_pool: "train" or "test"
        :return: prepared batch for tensor input
        """
        self.verify_data_pool_is_valid(data_pool)
        self.roll_data_for_batch(batch_size, data_pool)
        batch_input = self.input[data_pool][:batch_size]
        batch_y = self.y[data_pool][:batch_size]
        batch = self.data_utils.prepare_batch(batch_input, batch_y)
        yield batch

    @staticmethod
    def verify_data_pool_is_valid(data_pool):
        """
        data pool assertion
        :param data_pool: string
        :return:
        """
        assert data_pool == 'train' or data_pool == 'test'

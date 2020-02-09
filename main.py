from data_loader.data_generator import DataGenerator
from models.ssd_model import SSDModel
from runners.runner import Runner
from figures.figure import Figure
from deployers.tflite_converter import TFLiteConverter
from utils.config import get_config
from utils.utils import create_dirs
from utils.logger import Logger
from utils.utils import add_sys_paths
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior


def main():
    config = get_config()
    create_dirs([config.summary_dir, config.checkpoint_dir, config.figure_dir])
    add_sys_paths(config.add_paths_to_system_PATH_var)
    sess = tf.Session()
    model = SSDModel(config)
    data = DataGenerator(config, model)
    logger = Logger(sess, config)
    figure = Figure(config, logger)
    runner = Runner(sess, model, data, config, logger, figure)
    tflite_converter = TFLiteConverter(sess, model, config)
    model.load(sess)
    runner.train_and_test()
    tflite_converter.convert()


if __name__ == '__main__':
    main()

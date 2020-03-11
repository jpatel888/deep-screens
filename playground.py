from data_loader.data_generator import DataGenerator
from figures.figure import Figure
from utils.config import get_config
from utils.logger import Logger
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


config = get_config()
data = DataGenerator(config)
imgs, labels = next(data.next_batch(2))
img, label = imgs[0], labels[0]
sess = tf.Session()
logger = Logger(sess, config)
figure = Figure(config, logger)
figure.draw_test(img, label=label)

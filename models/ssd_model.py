from base.base_model import BaseModel
from models.ssd_defect_detector import FeatureMapper
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class SSDModel(BaseModel):
    def __init__(self, config):
        super(SSDModel, self).__init__(config)
        self.config = config
        self.saver = None
        self.input = None
        self.y = None
        self.feature_mapper = None
        self.model_output = None
        self.loss = None
        self.optimizer = None
        self.train_step = None
        self.layers = []
        self.define_input_placeholders()
        self.define_model()
        self.define_loss()
        self.define_optimizer()

    def define_input_placeholders(self):
        """
        defines input image placeholder, feature map placeholders will be generated with loss
        :return: N/A
        """
        input_shape = [self.config.batch_size] + self.config.input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape)

    def build_model(self):
        """
        Builds and calls feature map layer for output
        :return: N/A
        """
        self.feature_mapper = FeatureMapper(self.config)
        self.model_output = self.feature_mapper(self.input)

    def define_loss(self):
        """
        Define y placeholder(s) with feature map output as shape ref.
        :return: N/A
        """
        self.y = tf.placeholder(tf.float32, shape=self.model_output.shape)
        self.loss = None  # TODO: Define loss for multiple feature maps

    def define_optimizer(self):
        """
        Define an optimizer to minimize loss
        :return:
        """
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        """
        Initialize saver to store last n iterations of training, n is defined in config
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.num_models_to_save)

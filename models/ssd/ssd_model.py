from base.base_model import BaseModel
from models.ssd.ssd_defect_detector import FeatureMapper
from models.ssd.ssd_post_processor import PostProcessor
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
        self.post_processor = None
        self.model_output = None
        self.post_processed = None
        self.sigmoid_cross_entropy_loss = None
        self.l2_loss = None
        self.loss = None
        self.optimizer = None
        self.train_step = None
        self.layers = []
        self.define_input_placeholders()
        self.define_model()
        self.define_loss()
        self.define_optimizer()
        self.init_saver()

    def get_tflite_input_output_tensors(self):
        return [self.input], [self.post_processed]

    def define_input_placeholders(self):
        """
        defines input image placeholder, feature map placeholders will be generated with loss
        :return: N/A
        """
        input_shape = [self.config.model.batch_size] + self.config.model.input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape)

    def define_model(self):
        """
        Builds and calls feature map layer for output
        :return: N/A
        """
        self.feature_mapper = FeatureMapper(self.config)
        self.post_processor = PostProcessor(self.config)
        self.model_output = self.feature_mapper(self.input)
        self.post_processed = self.post_processor(self.model_output)

    def define_loss(self):
        """
        Add loss mask where has_defect != 1
        Define y placeholder(s) with feature map output as shape ref.
        :return: N/A
        """
        self.y = tf.placeholder(tf.float32, shape=self.model_output.shape)
        y_cross_entropy, y_l2 = self.y[:, :, :, :5], self.y[:, :, :, 5:]
        self.sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_cross_entropy,
            logits=self.post_processed[:, :, :, :5])
        difference = self.post_processed[:, :, :, 5:] - y_l2
        self.l2_loss = tf.reduce_sum(((difference * difference) / 2) * self.get_cost_mask())
        self.loss = (self.config.loss.l2_scalar * self.l2_loss) + (self.config.loss.sigmoid_scalar * tf.reduce_sum(self.sigmoid_cross_entropy_loss))

    def get_cost_mask(self):
        has_defect = self.y[:, :, :, 0]
        return tf.expand_dims(has_defect, axis=-1)

    def define_optimizer(self):
        """
        Define an optimizer to minimize loss
        :return:
        """
        self.optimizer = tf.train.AdamOptimizer(self.config.model.learning_rate, beta1=self.config.model.beta1, beta2=self.config.model.beta2)
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        """
        Initialize saver to store last n iterations of training, n is defined in config
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.model.num_models_to_save)

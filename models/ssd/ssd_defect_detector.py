import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class FeatureMapper(tf.keras.layers.Layer):
    def __init__(self, config):
        self.config = config
        self.layers = []
        self.c = (self.config.model.conv_filter_size, self.config.model.conv_filter_size)
        self.m = (self.config.model.max_pooling_size, self.config.model.max_pooling_size)
        super(FeatureMapper, self).__init__()

    def build(self, input_shape):
        """
        Build model layers sequentially and store as class variables
        :param input_shape: irrelevant since model is purely convolutional
        :return:
        """
        for kernel_size in self.config.model.conv_filters:
            self.layers.append(tf.keras.layers.Conv2D(kernel_size, self.c, activation=tf.nn.relu))
            self.layers.append(tf.keras.layers.MaxPooling2D(self.m, self.m))
            self.layers.append(tf.keras.layers.Dropout(self.config.model.dropout_rate, seed=self.config.model.dropout_seed))
        self.layers.append(tf.keras.layers.Conv2D(self.config.model.conv_filters[-1], self.c, activation=None))

    def call(self, inputs):
        """
        run inputs through class layers and return outputs
        :return:
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

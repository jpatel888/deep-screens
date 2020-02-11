import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class FeatureMapper(tf.keras.layers.Layer):
    def __init__(self, config):
        self.config = config
        self.layers = []
        self.c = (self.config.conv_filter_size, self.config.conv_filter_size)
        self.m = (self.config.max_pooling_size, self.config.max_pooling_size)
        super(FeatureMapper, self).__init__()

    def __call__(self, *args, **kwargs):
        super(FeatureMapper, self).__call__(args, kwargs)

    def build(self, input_shape):
        """
        Build model layers sequentially and store as class variables
        :param input_shape: irrelevant since model is purely convolutional
        :return:
        """
        out_depth = (self.config.num_classes + 5) * len(self.config.anchor_boxes)
        for kernel_size in self.config.conv_filters:
            self.layers.append(tf.keras.layers.Conv2D(kernel_size, self.c, activation=tf.nn.relu))
            self.layers.append(tf.keras.layers.MaxPooling2D(self.m, self.m))
            self.layers.append(tf.keras.layers.Dropout(self.config.dropout_rate))
        self.layers.append(tf.keras.layers.Conv2D(out_depth, self.c))

    def call(self, inputs):
        """
        run inputs through class layers and return outputs
        :return:
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
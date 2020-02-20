import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf
tf.sigmoid

class PostProcessor(tf.keras.layers.Layer):
    def __init__(self, config):
        self.config = config
        super(PostProcessor, self).__init__(config)

    def call(self, model_output):
        """
        Will first need to sigmoid [:, :, :7] so l2 can be applied to whole thing
        :param model_output: 16 x 9 x 9 grid
        :return:
        """
        first_seven, wh = model_output[:, :, :, :7], model_output[:, :, :, 7:]
        first_seven = tf.sigmoid(first_seven)
        return tf.concat((first_seven, wh), axis=-1)

    def output_ops(self, feature_mapper_output):
        """
        s: sigmoid mask
        x: (pixels_per_box_x * box_index_x) + (pixels_per_box_x * sigmoid(x_output)
        y: (pixels_per_box_y * box_index_y) + (pixels_per_box_y * sigmoid(y_output)
        w: (anchor_w * exp(w_output) * 32)
        h: (anchor_h * exp(h_output) * 32)
        :param feature_mapper_output: output grid
        :return:
        """
        model_output_depth = feature_mapper_output.shape[-1]
        self.bounding_box_xy_mask_placeholder = tf.placeholder(tf.float32, shape=[model_output_depth])
        self.bounding_Box_wh_mask_placeholder = tf.placeholder(tf.float32, shape=[model_output_depth])
        self.l2_mask = self.bounding_box_xy_mask_placeholder + self.bounding_Box_wh_mask_placeholder
        self.sigmoid_mask = 1 - self.l2_mask
        return tf.sigmoid(feature_mapper_output)

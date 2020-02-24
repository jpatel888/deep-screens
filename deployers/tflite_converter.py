import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TFLiteConverter:
    def __init__(self, sess, model, config):
        self.sess = sess
        self.config = config
        self.model = model

    def convert(self):
        """
        Uses in/out tensors defined in model.get_tflite_io_tensors() to create
            deployable tflite file
        :return: N/A
        """
        if not self.config.do_deploy:
            return print("Skipping Deployment")
        in_tensors, out_tensors = self.model.get_tflite_input_output_tensors()
        converter = tf.lite.TFLiteConverter.from_session(self.sess, in_tensors, out_tensors)
        tflite_model = converter.convert()
        open(self.config.exp_name + ".tflite", "wb").write(tflite_model)

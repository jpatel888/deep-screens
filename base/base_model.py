import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_current_epoch()

    def save(self, sess):
        if self.config.run.do_save_model:
            #print("Saving model...")
            self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
            #print("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint and self.config.run.do_restore_model:
            #print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            #print("Model Loaded")

    def init_current_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        raise NotImplementedError

    def define_model(self):
        raise NotImplementedError

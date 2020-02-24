import tensorflow as tf


class BaseRun:
    def __init__(self, sess, model, data, config, logger, figure):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.figure = figure
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train_and_test(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch) if self.config.do_train else print("Skipping Training")
            self.test_epoch(cur_epoch) if self.config.do_test else print("Skipping Training")
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, epoch_num):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def test_epoch(self, epoch_num):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

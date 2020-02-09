from base.base_run import BaseRun
from tqdm import tqdm
import numpy as np


class Runner(BaseRun):
    def __init__(self, sess, model, data, config, logger, figure):
        super(Runner, self).__init__(sess, model, data, config, logger, figure)

    def train_epoch(self, epoch_num):
        """
        Runs training and logging on train images
        :param epoch_num: current iteration of all training data passed
        :return:
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss, input_image, label, logit = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': loss
        }
        if epoch_num % 20 == 0:
            self.figure.draw_figure((input_image, label, logit), epoch_num, "train")
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def test_epoch(self, epoch_num):
        """
        Runs testing and logging on test images
        :param epoch_num: current iteration of all training data passed
        :return:
        """
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss, input_image, label, logit = self.test_step()
            losses.append(loss)
        loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'test_loss': loss
        }
        if epoch_num % 20 == 0:
            self.figure.draw_figure((input_image, label, logit), epoch_num, "test")
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        """
        Runs one step of training
        :return: loss, input image, and model output
        """
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, 'train'))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        optimizer, loss, results = self.sess.run([self.model.train_step,
                                              self.model.cross_entropy,
                                              self.model.accuracy],
                                             feed_dict=feed_dict)
        return loss, batch_x[0], batch_y[0], results[0]

    def test_step(self):
        """
        Runs one step of testing
        :return: loss, input_image, model output
        """
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, 'test'))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        loss, results = self.sess.run([self.model.decoded_image_loss,
                                       self.model.decoded_image],
                                      feed_dict=feed_dict)
        return loss, batch_x[0], batch_y[0], results[0]

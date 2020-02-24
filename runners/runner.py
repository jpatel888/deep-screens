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
        loop = tqdm(range(self.config.num_iter_per_train_epoch))
        losses, l2_losses, sigmoid_losses = [], [], []
        for _ in loop:
            loss, l2_loss, sigmoid_loss, input_image, label, logit = self.train_step()
            losses.append(loss)
            l2_losses.append(l2_loss)
            sigmoid_losses.append(sigmoid_loss)
        loss = np.mean(losses)
        l2_loss = np.mean(l2_losses)
        sigmoid_loss = np.mean(sigmoid_losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'l2_loss': l2_loss,
            'sigmoid_loss': sigmoid_loss
        }
        if epoch_num % 1 == 0:
            self.figure.draw_figure((input_image, label, logit), cur_it, summarizer="train", tag="images")
        self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="train")
        self.model.save(self.sess)

    def test_epoch(self, epoch_num):
        """
        Runs testing and logging on test images
        :param epoch_num: current iteration of all training data passed
        :return:
        """
        loop = tqdm(range(self.config.num_iter_per_test_epoch))
        losses, l2_losses, sigmoid_losses = [], [], []
        for _ in loop:
            loss, l2_loss, sigmoid_loss, input_image, label, logit = self.test_step()
            losses.append(loss)
            l2_losses.append(l2_loss)
            sigmoid_losses.append(sigmoid_loss)
        loss = np.mean(losses)
        l2_loss = np.mean(l2_losses)
        sigmoid_loss = np.mean(sigmoid_losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'l2_loss': l2_loss,
            'sigmoid_loss': sigmoid_loss
        }
        if epoch_num % 1 == 0:
            self.figure.draw_figure((input_image, label, logit), cur_it, summarizer="test", tag="images")
        self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="test")
        self.model.save(self.sess)

    def train_step(self):
        """
        Runs one step of training
        :return: loss, input image, and model output
        """
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, 'train'))
        feed_dict = {self.model.input: batch_x, self.model.y: batch_y}
        optimizer, loss, l2_loss, sigmoid_loss, results = self.sess.run([self.model.train_step,
                                                                         self.model.loss,
                                                                         self.model.l2_loss,
                                                                         self.model.sigmoid_cross_entropy_loss,
                                                                         self.model.post_processed],
                                                                        feed_dict=feed_dict)
        return loss, l2_loss, sigmoid_loss, batch_x[0], batch_y[0], results[0]

    def test_step(self):
        """
        Runs one step of testing
        :return: loss, input_image, model output
        """
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, 'test'))
        feed_dict = {self.model.input: batch_x, self.model.y: batch_y}
        loss, l2_loss, sigmoid_loss, results = self.sess.run([self.model.loss,
                                                              self.model.l2_loss,
                                                              self.model.sigmoid_cross_entropy_loss,
                                                              self.model.post_processed],
                                                             feed_dict=feed_dict)
        return loss, l2_loss, sigmoid_loss, batch_x[0], batch_y[0], results[0]

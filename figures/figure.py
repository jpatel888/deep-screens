from base.base_figure import BaseFigure
import numpy as np


class Figure(BaseFigure):
    """
    Figure:
    want ot generate a pic of new with boxes (in: (), out: ())
    want ot generate a pic of sigmoids for whether or not there's a defect

    Cost:
    want to generate a cost mask from labels:
        if (has_defect): 1 for everything
        else: 1 for has_defect, 0 for ddddxywh

    Model:
    want to pass data through convnet until 16x9x9 output

    Logger:
    want to, at every global step, log:
        image of sigmoids (predicted vs actual, train & val)
        image of current with boxes, vs baseline with boxes (train & validation)
        cost (train and validation

    Data Loader:
    want to pull a batch of images and grid of labels for L2 Loss
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def draw_test(self, img, labels):
        pass

    def draw_figure(self, data, step, summarizer="train", tag=""):
        """
        TODO: Add Box Draw Visualization:
        :param data: images
        :param step: global step number for training/testing
        :param summarizer: "train" or "test"
        :param tag: any added details as string
        :return:
        """
        input_image, label_grid, logit_grid = data
        # TODO: create three images from the
        #       two (baseline and current):
        #           1. Baseline
        #           2. Current (with label_grid)
        #           3. Current (with logit_grid)
        #       append the images together by width
        if len(list(data.shape)) == 4:
            summaries_dict = {tag: data}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)
        elif len(list(data.shape)) == 3:
            summaries_dict = {tag: np.expand_dims(data, axis=0)}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)

from base.base_figure import BaseFigure
import numpy as np


class Figure(BaseFigure):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def draw_figure(self, data, step, summarizer="train", tag=""):
        """
        TODO: Add Box Draw Visualization
        :param data: images
        :param step: global step number for training/testing
        :param summarizer: "train" or "test"
        :param tag: any added details as string
        :return:
        """
        input_image, label, logit = data
        if len(list(data.shape)) == 4:
            summaries_dict = {tag: data}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)
        elif len(list(data.shape)) == 3:
            summaries_dict = {tag: np.expand_dims(data, axis=0)}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)

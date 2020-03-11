from base.base_figure import BaseFigure
from figures.image import Image
import numpy as np


class Figure(BaseFigure):
    """
    Data Loader:
    want to pull a batch of images and grid of labels for L2 Loss
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def tf_log(self, data, step, summarizer, tag):
        if len(list(data.shape)) == 4:
            summaries_dict = {tag: data}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)
        elif len(list(data.shape)) == 3:
            summaries_dict = {tag: np.expand_dims(data, axis=0)}
            self.logger.summarize(step, summarizer=summarizer, summaries_dict=summaries_dict)

    def draw_figure(self, data, step, summarizer="train", tag=""):
        input_image, label_grid, logit_grid = data
        image_obj = Image(self.config, input_image, label_grid, logit_grid)
        data = image_obj.get_log_image()
        self.tf_log(data, step, summarizer, "boxes")

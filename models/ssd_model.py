from base.base_model import BaseModel
import tensorflow as tf


class SSDDefectDetector(BaseModel):
    def __init__(self, config):
        super(SSDDefectDetector, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        pass

    def init_saver(self):
        pass


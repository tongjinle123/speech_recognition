from .base_class import BaseClass
from .base_config import ConfigDict
import torch as t


class BaseModel(BaseClass, t.nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def save(self, path):
        saves = self.state_dict()
        t.save(saves, path)
        print(f'\nmodel saved to {path}')

    def load(self, path):
        saves = t.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(saves)

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(example_model_config='example_config')
        return config

    def forward(self, inputs):
        raise NotImplementedError

    def iterate(self, inputs):
        raise NotImplementedError

    def cal_metric(self, inputs):
        raise NotImplementedError


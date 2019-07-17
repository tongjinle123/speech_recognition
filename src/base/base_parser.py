from .base_class import BaseClass
from .base_config import ConfigDict
from torch.utils.data import Dataset, DataLoader


class BaseParser(BaseClass):
    def __init__(self, config):
        super(BaseParser, self).__init__()
        self.config = config

    @classmethod
    def load_default_config(cls) -> ConfigDict:
        config = ConfigDict()
        config.add(example_parser_config='example_parser_config')
        return config

    def parse(self):
        raise NotImplementedError

    def build_dataset(self, set):
        raise NotImplementedError

    def build_iters(self):
        raise NotImplementedError

    @property
    def collate_fn(self):
        def collate_fn(batch):
            return batch
        return collate_fn


class Loader(DataLoader):
    def __init__(self, *args, **kwargs, ):
        super(Loader, self).__init__(args, kwargs)

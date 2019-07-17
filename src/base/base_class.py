from .base_config import ConfigDict


class BaseClass:

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        #config.a = 1
        return config




from .base_class import BaseClass
from .base_config import ConfigDict
import os
import torch as t
from torch.utils.tensorboard import SummaryWriter
import datetime
from .utils import load_module
from src import models
from src import parsers


def get_time():
    return (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d_%H%M_%S")


class BaseSolver(BaseClass):
    def __init__(self, config):
        super(BaseSolver, self).__init__()
        self.config = config

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(example_config='example_config')
        config.add(from_ckpt=None)
        return config

    def _init_experiment(self):
        #name = input('input experiment name:')
        name=''
        if name == '':
            self.config.add(exp_name=get_time())
        else:
            self.config.add(exp_name=name)
        exp_folder = os.path.join(self.config.experiment_root, self.config.exp_name)
        print(f'experiment initialized in {exp_folder}')
        assert not os.path.exists(exp_folder)
        os.mkdir(exp_folder)
        self.config.save(os.path.join(exp_folder, 'config.yaml'))
        self.summary_writer = SummaryWriter(exp_folder)
        self.global_epoch = 0
        self.global_step = 0

    def _init_model(self):
        Model, _ = load_module(models, self.config.model_name)
        self.model = Model(self.config)
        if self.config.from_ckpt is not None:
            self.model.load(self.config.from_ckpt)
            print(f'\nmodel loaded from {self.config.from_ckpt}')

    def _init_parser(self):
        Parser, _ = load_module(parsers, self.config.parser_name)
        self.parser = Parser(self.config)
        self.train_iter, self.dev_iter, self.test_iter = self.parser.build_iters()

    def _init_optimizer(self):
        assert self.model
        self.optimizer = t.optim.Adam(self.model.parameters())

    def train_from_scrach(self):
        self._init_experiment()
        self._init_parser()
        self._init_model()
        self._init_optimizer()
        self._train()

    def train_from_ckpt(self):
        self._init_experiment()
        self._init_parser()
        self._init_model()
        self._init_optimizer()
        self._train()

    def _train(self):
        for i in range(self.config.num_epoch):
            self._train_epoch()
            self.global_epoch += 1

    def _train_epoch(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


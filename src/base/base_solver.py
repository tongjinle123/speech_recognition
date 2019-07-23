from .base_class import BaseClass
from .base_config import ConfigDict
import os
import torch as t
from torch.utils.tensorboard import SummaryWriter
import datetime
from .utils import load_module
from src import models
from src import parsers
from tqdm import tqdm
from src.base.utils import MetricsManager



def get_time():
    return (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d_%H%M_%S")


class BaseSolver(BaseClass):
    def __init__(self, config):
        super(BaseSolver, self).__init__()
        self.config = config

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(
            experiment_root='experiments/',
            from_ckpt=None,  # f'{expname}_{epoch}_{step}'
            lr=1e-3,
            num_epoch=200,
            warm_up=25000,
            factor=0.8,
            smoothing=0.0,
            log_every_iter=100,
            eval_every_iter=5000,
            save_every_iter=5000,
            device_id=0
        )
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

    def _init_model(self, model_path=None):
        Model, _ = load_module(models, self.config.model_name)
        self.model = Model(self.config, self.vocab)
        if model_path is not None:
            self.model.load(model_path)
        self.model.cuda()

    def _init_parser(self):
        Parser, _ = load_module(parsers, self.config.parser_name)
        self.parser = Parser(self.config)
        self.vocab = self.parser.vocab
        self.train_iter, self.dev_iter, self.test_iter = self.parser.build_iters()

    def _init_optimizer(self, optimizer_path=None):
        assert self.model
        self.optimizer = t.optim.Adam(self.model.parameters())

    def train_from_scrach(self):
        self._init_parser()
        self._init_model()
        self._init_optimizer()
        self._train()

    def train_from_ckpt(self, from_ckpt):
        tmp_exp_name = self.config.exp_name
        tmp_experiment_root = self.config.experiment_root
        model_path, optimizer_path, epoch, step, old_yaml_path = self._extract_ckptname(from_ckpt)
        self.config = ConfigDict()
        self.config.load(old_yaml_path)
        self.config.exp_name = tmp_exp_name
        self.config.experiment_root = tmp_experiment_root
        self.config.save(os.path.join(self.config.experiment_root, self.config.exp_name, 'config.yaml'))
        self.global_epoch = int(epoch)
        self.global_step = int(step)
        self._init_parser()
        self.train_iter, self.dev_iter, self.test_iter = self.parser.build_iters()
        self._init_model(model_path)
        self._init_optimizer(optimizer_path)
        self._train()

    def _build_ckptname(self, experiment_name, epoch, step):
        model_path = os.path.join(self.config.experiment_root, experiment_name, f'{int(epoch)}_{int(step)}.model')
        optimizer_path = os.path.join(self.config.experiment_root, experiment_name, f'{int(epoch)}_{int(step)}.opt')
        return model_path, optimizer_path

    def _extract_ckptname(self, from_ckpt):
        day, hour, sec, epoch, step = from_ckpt.split('_')
        experiment_name = '_'.join([day, hour, sec])
        model_path = os.path.join(self.config.experiment_root, experiment_name, f'{int(epoch)}_{int(step)}.model')
        optimizer_path = os.path.join(self.config.experiment_root, experiment_name, f'{int(epoch)}_{int(step)}.opt')
        yaml_path = os.path.join(self.config.experiment_root, experiment_name, 'config.yaml')
        return model_path, optimizer_path, epoch, step, yaml_path

    def _save(self):
        model_path, optimizer_path = self._build_ckptname(
            self.config.exp_name, self.global_epoch, self.global_step)
        self.model.save(model_path)
        self.optimizer.save(optimizer_path)

    def get_model(self, from_ckpt):
        model_path, optimizer_path, epoch, step, old_yaml_path = self._extract_ckptname(from_ckpt)
        self.config = ConfigDict()
        self.config.load(old_yaml_path)
        self._init_parser()
        self._init_model(model_path)
        return self.parser, self.model

    def _train(self):
        for i in range(self.config.num_epoch):
            self._train_epoch()
            self.global_epoch += 1

    def _train_epoch(self):
        self.model.train()
        max_len = 0
        train_bar = tqdm(iterable=self.train_iter, leave=True, total=len(self.train_iter))
        average_loss = 0
        for i, data in enumerate(train_bar):

            data = {i: v.cuda() for i, v in data.items()}
            metrics, _ = self.model.iterate(data, optimizer=self.optimizer, is_train=True)
            lr = self.optimizer.rate()
            self.summary_writer.add_scalar('lr', lr, self.global_step)

            if self.global_step % self.config.log_every_iter == 0 and self.global_step != 0:
                self.summarize(metrics, 'train/')

            self.global_step += 1
            if self.global_step % self.config.eval_every_iter == 0 and self.global_step != 0:
                self.evaluate(self.dev_iter, 'dev/')

            if self.global_step % self.config.save_every_iter == 0 and self.global_step != 0:
                self._save()
            le = data['wave'].size(1)
            if le > max_len:
                max_len = le
            average_loss += metrics.loss.item()
            desc = f'ep: {self.global_epoch}, lr: {round(self.optimizer._rate, 5)}, ml: {max_len}, loss: {round(average_loss / (i+1), 3)}, cl:{round(metrics.loss.item(), 3)} cer: {round(metrics.cer.item(), 3)}'
            train_bar.set_description(desc)
        # self.save_ckpt(metrics[self.reference[1:]])
        # self.evaluate(self.test_iter, 'test/')

    def evaluate(self, dev_iter, prefix='dev/'):
        print(f'\nEvaluating')
        self.model.eval()
        dev_metric_manager = MetricsManager()
        dev_bar = tqdm(dev_iter, leave=True, total=len(dev_iter), disable=True)
        average_loss = 0
        average_score = 0
        with t.no_grad():
            for data in dev_bar:
                data = {i:v.cuda() for i, v in data.items()}
                metrics, _ = self.model.iterate(data, is_train=False)
                dev_metric_manager.update(metrics)
                desc = f'Valid-loss: {round(metrics.loss.item(), 3)}, cer: {round(metrics.cer.item(), 3)}'
                dev_bar.set_description(desc)
                average_loss += metrics.loss.item()
                average_score += metrics.cer.item()

            print(f'\nValid, average_loss: {average_loss / len(dev_iter)}, average_score: {average_score / len(dev_iter)}\n')
            report = dev_metric_manager.report_cum()
            report = dev_metric_manager.extract(report)
            self.summarize(report, 'dev/')
        self.model.train()

    def summarize(self, pack, prefix='train/'):
        # print(f'\nsummarizing in {self.global_step}')
        for i in pack:
            tmp_prefix = prefix + i
            self.summary_writer.add_scalar(tmp_prefix, pack[i].detach().cpu().numpy(), self.global_step)


from src.base.base_solver import BaseSolver
from src.base.base_config import ConfigDict
from .utils.optimizer import AdamW, NoamOpt, RAdam
from torch.optim import Adam

class Solver(BaseSolver):
    def __init__(self, config):
        super(Solver, self).__init__(config)

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(
            experiment_root='experiments/',
            from_ckpt=None,  # f'{expname}_{epoch}_{step}'
            lr=1e-3,
            num_epoch=200,
            warm_up=20000,
            factor=0.8,
            log_every_iter=100,
            eval_every_iter=5000,
            save_every_iter=5000,
            device_id=0
        )
        return config

    def _init_optimizer(self, optimizer_path=None):

        optimizer = AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = NoamOpt(
            self.config.model_size, factor=self.config.factor, warmup=self.config.warm_up, optimizer=optimizer)

        if optimizer_path is not None:
            self.optimizer.load(optimizer_path)
            

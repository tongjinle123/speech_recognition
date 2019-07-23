import fire
from src import models
from src import solvers
from src import parsers
from src.base.utils import load_module
from src.base.base_config import ConfigDict
import torch as t
import warnings
warnings.filterwarnings('ignore')


def train(**kwargs):
    model_name = 'DynamicCNNTransformerCTC'
    parser_name = 'ParserAishell'
    solver_name = 'Solver'
    if not 'model_name' in kwargs:
        kwargs['model_name'] = model_name
    else:
        print(kwargs['model_name'])
    if not 'parser_name' in kwargs: kwargs['parser_name'] = parser_name
    if not 'solver_name' in kwargs: kwargs['solver_name'] = solver_name
    print(kwargs['parser_name'])
    _, model_config = load_module(models, kwargs['model_name'])
    _, parser_config = load_module(parsers, kwargs['parser_name'])
    Solver, solver_config = load_module(solvers, kwargs['solver_name'])

    configs = ConfigDict()
    configs.combine(model_config)
    configs.combine(parser_config)
    configs.combine(solver_config)
    configs.add(model_name=kwargs['model_name'])
    configs.add(solver_name=kwargs['solver_name'])
    configs.add(parser_name=kwargs['parser_name'])
    configs.update(kwargs)
    configs.show

    t.cuda.set_device(configs.device_id)
    solver = Solver(configs)
    print(solver.config.model_name)
    print(solver.config.parser_name)
    solver._init_experiment()
    if configs.from_ckpt is not None:
        solver.train_from_ckpt(configs.from_ckpt)
    else:
        solver.train_from_scrach()

def get_model(from_ckpt, solver_name):
    Solver, solver_config = load_module(solvers, solver_name)
    solver = Solver(solver_config)
    parser, model = solver.get_model(from_ckpt)
    return parser, model


if __name__ == '__main__':
    fire.Fire()


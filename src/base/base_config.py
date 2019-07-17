import yaml


class ConfigDict:
    def __init__(self, **kwargs):
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value

    def add(self, *args, **kwargs):
        for key, value in kwargs.items():
            self.config[key] = value
        if args != ():
            for key, value in args[0].items():
                self.config[key] = value

    def update(self, *args, **kwargs):
        for key, value in kwargs.items():
            assert key in self.config
        self.config.update(kwargs)
        if args != ():
            for key, value in args[0].items():
                assert key in self.config
                self.config[key] = value

    def combine(self, other):
        self.add(other.config)

    @property
    def show(self):
        print(f'config:')
        for key, value in self.config.items():
            print(f'\t{key}: {value}')

    def __getattr__(self, item):
        return self.config[item]

    def save(self, path):
        yaml.dump(self.config, open(path, 'w'))

    def load(self, path):
        self.config = yaml.safe_load(open(path))

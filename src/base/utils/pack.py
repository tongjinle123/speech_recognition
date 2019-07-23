

class Pack(dict):
    """
    Pack
    """
    def __init__(self):
        super(Pack, self).__init__()

    def __getattr__(self, name):
        return self.get(name)

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def cuda(self):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda() for x in v)
            else:
                pack[k] = v.cuda()
        return pack

import multiprocessing

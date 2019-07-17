
def load_module(package, name):
    _class = getattr(package, name)
    default_config = _class.load_default_config()
    return _class, default_config

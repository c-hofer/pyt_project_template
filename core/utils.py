class Factory:
    def __init__(self, provider):
        self._provider = provider

    def __call__(self, arg):
        if isinstance(arg, str):
            return getattr(self._provider, arg)()
        elif isinstance(arg, tuple):
            id, kwargs = arg
            return getattr(self._provider, id)(**kwargs)
        else:
            raise ValueError()
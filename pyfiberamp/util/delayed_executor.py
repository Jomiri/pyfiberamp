

class DelayedExecutor:
    def __init__(self):
        self.funcs_and_args = []

    def add_func(self, func, args):
        self.funcs_and_args.append((func, args))

    def execute(self):
        for func, args in self.funcs_and_args:
            func(*args)

    def reset(self):
        self.funcs_and_args = []

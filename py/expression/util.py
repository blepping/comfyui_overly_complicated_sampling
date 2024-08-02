import itertools

try:
    import torch
except ImportError:
    # To facilitate testing.
    class torch:
        class Tensor:
            pass


class WrapGenerator:
    def __init__(self, g):
        self.g = g
        self._value = None
        self.ready = False

    @property
    def value(self):
        if not self.ready:
            raise ValueError("Value not ready")
        return self._value

    def __iter__(self):
        self._value = yield from self.g
        self.ready = True
        return self._value


def split_iterable(seq, pred):
    it = iter(seq)
    while True:
        toks = tuple(itertools.takewhile(pred, it))
        if toks == ():
            break
        yield toks

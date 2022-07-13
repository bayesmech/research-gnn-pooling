import abc

import numpy as np


class HyperParameter(abc.ABC):

    def __init__(self, name, value=0.):
        self.name = name
        self.value = value

    @abc.abstractmethod
    def update(self, epoch) -> None:
        """
        Updates done at every single epoch
        :param epoch:
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)


class ExponentiallyDecayingHyperParameter(HyperParameter):

    def __init__(self, name, value, exp_multiplier, lowest_value=0.):
        super().__init__(name, value)
        self.exp_multiplier = exp_multiplier
        self.lowest_value = lowest_value
        self.initial_value = value

    def update(self, epoch) -> None:
        self.value = max(self.lowest_value, self.initial_value * np.exp(-self.exp_multiplier * epoch))


class StaticHyperParameter(HyperParameter):

    def __init__(self, name, value):
        super().__init__(value, name)

    def update(self, epoch) -> None:
        pass

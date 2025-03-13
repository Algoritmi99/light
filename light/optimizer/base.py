import copy

from light import Linear, Sequential
from light.loss.base import Loss
from light.nn.base import Module


class Optimizer:
    def __init__(self, network: Module, loss: Loss, learning_rate: float):
        self.network = copy.deepcopy(network)
        self.loss = loss
        self.learning_rate = learning_rate
        self._update_factors: dict = {}

    def step(self, y_true, y_pred):
        self._calc_update_factors(y_true, y_pred)
        self.__update()
        self._update_factors = {}

    def _calc_update_factors(self, y_true, y_pred):
        raise NotImplementedError

    def __update(self):
        assert self._update_factors
        for attr_name in dir(self.network):
            attr = getattr(self.network, attr_name)
            if isinstance(attr, Linear):
                attr.weight += self._update_factors[id(attr.weight)]
                attr.bias += self._update_factors[id(attr.bias)]
            if isinstance(attr, Sequential):
                for layer in attr.layers:
                    if isinstance(layer, Linear):
                        layer.weight += self._update_factors[id(layer.weight)]
                        layer.bias += self._update_factors[id(layer.bias)]

    def __call__(self, y_true, y_pred):
        self.step(y_true, y_pred)

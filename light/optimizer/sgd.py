from overrides import override

from light import Linear, Sequential
from light.optimizer.base import Optimizer


class SGD(Optimizer):
    @override
    def _calc_update_factors(self, y_true, y_pred):
        loss_error = self.loss.gradient(y_true, y_pred)
        self.network.backward(loss_error)
        for attr_name in dir(self.network):
            attr = getattr(self.network, attr_name)
            if isinstance(attr, Linear):
                self._update_factors[id(attr.weight)] = (-1 * self.learning_rate) * attr.grads['weight']
                self._update_factors[id(attr.bias)] = (-1 * self.learning_rate) * attr.grads['bias']
            if isinstance(attr, Sequential):
                for layer in attr.layers:
                    if isinstance(layer, Linear):
                        self._update_factors[id(layer.weight)] = (-1 * self.learning_rate) * layer.grads['weight']
                        self._update_factors[id(layer.bias)] = (-1 * self.learning_rate) * layer.grads['bias']

from overrides import override

import numpy as np

from light.nn.base import Module


class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self.layers: list[Module] = [module for module in args]

    def add_layer(self, layer: Module):
        self.layers.append(layer)

    @override
    def forward(self, arg: np.ndarray) -> np.ndarray:
        output = arg
        for layer in self.layers:
            output = layer.forward(output)
        return output

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        output = d_out
        for layer in reversed(self.layers):
            output = layer.backward(output)
        return output

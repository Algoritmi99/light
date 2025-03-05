from overrides import override

import numpy as np

from light.nn.module import ActivationFunction


def relu(x):
    return np.maximum(0, x)


class ReLU(ActivationFunction):
    """
        An implementation of the ReLU activation function.
    """

    def __init__(self):
        super().__init__(relu, None),

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        assert self.saved_input.shape == d_out.shape, "Shape of input and output error do not match."
        return np.array([0 if i <= 0 else 1 for i in self.saved_input]) * d_out

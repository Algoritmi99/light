import numpy as np
from overrides import override

from light.nn.base import ActivationFunction


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def softmax_prime(x):
    s = softmax(x).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


class Softmax(ActivationFunction):
    """
        An implementation of the Tanh activation function.
    """

    def __init__(self):
        super().__init__(softmax, softmax_prime)

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Passing the error because of combination with CrossEntropyLoss trick.
        :param d_out: propagated error from output.
        :return: pass through because of combination with CrossEntropyLoss trick.
        """
        assert self.saved_input.shape == d_out.shape, "Shape of input and output error do not match."
        return d_out


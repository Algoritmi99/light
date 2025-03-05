import numpy as np

from light.nn.module import ActivationFunction


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


class Tanh(ActivationFunction):
    """
        An implementation of the Tanh activation function.
    """

    def __init__(self):
        super().__init__(tanh, tanh_prime)

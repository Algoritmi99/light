import numpy as np

from light.nn.base import ActivationFunction


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))


class Sigmoid(ActivationFunction):
    """
        An implementation of the Sigmoid activation function.
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

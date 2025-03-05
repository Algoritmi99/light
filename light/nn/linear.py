import numpy as np
from overrides import override

from light.nn.module import Module


class Linear(Module):
    """
            Implements linear layer for Light,
            A fully connected layer of a neural network with a linear activation function.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, init_rand = True):
        super().__init__()
        self.grads = None
        self.in_features = in_features
        self.out_features = out_features
        self.bias = np.random.randn(out_features).astype(np.float32) if bias\
            else np.zeros(out_features).astype(np.float32)
        self.weight = np.random.randn(in_features, out_features).astype(np.float32) if init_rand \
            else np.zeros((in_features, out_features)).astype(np.float32)

    @override
    def forward(self, arg: np.ndarray) -> np.ndarray:
        assert arg.shape == (self.in_features, ), "Wrong input shape"
        self.saved_input = arg
        self.saved_output = np.dot(arg.T, self.weight) + self.bias
        return self.saved_output

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        assert d_out.shape == (self.out_features,), "Wrong gradient shape"

        d_weight = np.outer(self.saved_input, d_out)
        d_bias = d_out if self.bias is not None else None
        d_input = np.dot(self.weight, d_out)

        # Store gradients for optimization step
        self.grads = {'weight': d_weight, 'bias': d_bias}

        return d_input


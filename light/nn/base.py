import os
import pickle

import numpy as np
from overrides import override


def save_model(model, save_path: str, file_name: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + "/" + file_name + ".light", "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        agent = pickle.load(f)
        assert isinstance(agent, Module)
        return agent


class Module:
    """
    Serves as the base class of any light.nn module.
    """
    def __init__(self):
        """
        The default constructor for this module.
        keeps a place-holder for the latest input and output of the module.
        """
        self.saved_input = None
        self.saved_output = None

    def forward(self, arg: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass of the module.
        :param arg: The input to the module as a singular numpy array.
        :return: The output of the module as a numpy array.
        """
        raise NotImplementedError

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Implements the backward pass of the module.
        :param d_out: The ingoing error as a numpy array.
        :return: The input error as a numpy array.
        """
        raise NotImplementedError

    def save(self, save_path: str, file_name: str):
        """
        Saves the entire module as a .light file.
        :param save_path: The path to save the module to.
        :param file_name: The file name to save the module to.
        :return: null
        """
        save_model(self, save_path, file_name)

    def __call__(self, arg: np.ndarray) -> np.ndarray:
        return self.forward(arg)

class ActivationFunction(Module):
    """
        A common implementation for any activation function.
        The specific activation function together with its derivative must be
         provided when instantiating the class.
    """

    def __init__(self, activation_function, activation_function_derivative):
        super().__init__()
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    @override
    def forward(self, arg: np.ndarray) -> np.ndarray:
        self.saved_input = arg
        self.saved_output = self.activation_function(self.saved_input)
        return self.saved_output

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        assert self.saved_input.shape == d_out.shape, "Shape of input and output error do not match."
        return self.activation_function_derivative(self.saved_input) * d_out

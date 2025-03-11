import copy

import pandas as pd

from light import Optimizer, Module


class Trainer:
    """
    Class to run the training loop given an optimizer, which is instantiated with
    the corresponding model, learning rate and loss function.
    """
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def train(self, dataset: tuple[pd.DataFrame, pd.DataFrame], n_epochs: int) -> Module:
        """
        runs the training loop given a dataset and a number of epochs.
        :param dataset: tuple of datapoints and ground truth labels.
        :param n_epochs: number of epochs to train for.
        :return: trained model.
        """
        assert len(dataset[0]) == len(dataset[1]), "The datapoints and labels must have the same length."
        assert n_epochs > 0, "The number of epochs must be greater than 0."
        for epoch in range(n_epochs):
            for idx in range(len(dataset[0])):
                x = dataset[0].iloc[idx].to_numpy()
                y = dataset[1].iloc[idx].to_numpy()
                y_pred = self.optimizer.network.forward(x)
                self.optimizer.step(y, y_pred)
        return copy.deepcopy(self.optimizer.network)

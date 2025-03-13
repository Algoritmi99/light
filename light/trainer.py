import copy

import pandas as pd

from light import Optimizer, Module
from light.plotter import Plotter


class Trainer:
    """
    Class to run the training loop given an optimizer, which is instantiated with
    the corresponding model, learning rate and loss function.
    """
    def __init__(self, optimizer: Optimizer, plot=False):
        self.optimizer = optimizer
        self.plot = plot

    def train(self, dataset: tuple[pd.DataFrame, pd.DataFrame], n_epochs: int) -> Module:
        """
        runs the training loop given a dataset and a number of epochs.
        :param dataset: tuple of datapoints and ground truth labels.
        :param n_epochs: number of epochs to train for.
        :return: trained model.
        """
        assert len(dataset[0]) == len(dataset[1]), "The datapoints and labels must have the same length."
        assert n_epochs > 0, "The number of epochs must be greater than 0."

        plotter = Plotter(n_epochs, "Epoch", "Error") if self.plot else None

        for epoch in range(n_epochs):
            for idx in range(len(dataset[0])):
                x = dataset[0].iloc[idx].to_numpy()
                y = dataset[1].iloc[idx].to_numpy()
                y_pred = self.optimizer.network(x)
                if self.plot:
                    plotter.add_data(epoch, self.optimizer.loss(y, y_pred))

                self.optimizer.step(y, y_pred)

        if plotter is not None:
            plotter.make_plot("Error in Training")

        return copy.deepcopy(self.optimizer.network)

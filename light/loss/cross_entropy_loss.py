from overrides import override

import numpy as np

from light.loss.base import Loss


class CrossEntropyLoss(Loss):
    @override
    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10 ** -100))

    @override
    def gradient(self, y_true, y_pred):
        """
        combined with the softmax function
        :param y_true: the ground truth labels
        :param y_pred: the predicted labels
        :return: the gradient of the loss function.
        """
        return y_pred - y_true

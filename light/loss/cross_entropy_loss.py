from overrides import override

import numpy as np

from light.loss.loss import Loss


class CrossEntropyLoss(Loss):
    @override
    def loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10 ** -100))

    @override
    def gradient(self, y_true, y_pred):
        return y_true - y_pred

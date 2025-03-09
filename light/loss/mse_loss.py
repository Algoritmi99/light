from overrides import override

import numpy as np

from light.loss.base import Loss


class MSELoss(Loss):
    @override
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @override
    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

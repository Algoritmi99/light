class Loss(object):
    """
    The base class of all loss functions.
    """
    def loss(self, y_true, y_pred):
        """
        Compute the loss between y_true and y_pred.
        :param y_true: The ground truth tensor.
        :param y_pred: The predicted tensor.
        :return: The loss value.
        """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """
        Compute the gradient of the loss function.
        :param y_true: the ground truth tensor.
        :param y_pred: the predicted tensor.
        :return: the gradient of the loss function.
        """
        raise NotImplementedError
import numpy as np


class Dimension_Reduction:
    """
    Base class for dimensionality reduction techniques.
    """

    def __init__(self, n_components):
        """
        Initialize the dimensionality reduction algorithm.
        :param n_components: Number of components to retain
        """
        self.n_components = n_components

    def fit(self, X):
        """
        Compute the transformation based on input data.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the fit method")

    def transform(self, X):
        """
        Apply the learned transformation.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the transform method")

    def fit_transform(self, X):
        """
        Fit the model and transform the data.
        """
        self.fit(X)
        return self.transform(X)
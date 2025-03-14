class DimensionReduction:
    """
    Base class for dimensionality reduction techniques.
    """

    def __init__(self, n_components=None, variance_threshold=0.95):
        """
        Initialize the dimensionality reduction algorithm.
        :param n_components: Number of components to retain (if None, it will be chosen based on variance)
        :param variance_threshold: Minimum variance to retain when choosing n_components automatically
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def fit(self, X):
        raise NotImplementedError("Subclasses must implement the fit method")

    def transform(self, X):
        raise NotImplementedError("Subclasses must implement the transform method")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
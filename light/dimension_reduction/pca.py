import numpy as np
from overrides import override

from light.dimension_reduction.base import Dimension_Reduction


class PCA(Dimension_Reduction):
    def __init__(self, n_components):
        """
        Initialize PCA with the desired number of principal components.
        :param n_components: Number of principal components to retain
        """
        super().__init__(n_components)
        self.mean = None
        self.eigenvectors = None

    @override
    def fit(self, X):
        """
        Compute PCA on the dataset.
        :param X: Input data of shape (samples, features)
        """
        # Step 1: Standardize the data (zero mean, unit variance)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean  # Centering the data

        # Step 2: Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, sorted_indices[:self.n_components]]

    @override
    def transform(self, X):
        """
        Project the data onto the principal components.
        :param X: Input data of shape (samples, features)
        :return: Transformed data of shape (samples, n_components)
        """
        X_centered = X - self.mean  # Centering the data
        return np.dot(X_centered, self.eigenvectors)

    @override
    def fit_transform(self, X):
        """
        Fit PCA and transform the data in one step.
        :param X: Input data of shape (samples, features)
        :return: Transformed data of shape (samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

import numpy as np
from overrides import override

from light.dimension_reduction.base import DimensionReduction


class PCA(DimensionReduction):
    """
    Principal Component Analysis (PCA) implementation using NumPy.
    """

    def __init__(self, n_components=None, variance_threshold=0.95):
        super().__init__(n_components, variance_threshold)
        self.mean = None
        self.eigenvectors = None
        self.explained_variance_ratio = None

    @override
    def fit(self, X):
        """
        Compute PCA on the dataset.
        :param X: Input data of shape (samples, features)
        """
        # Step 1: Standardize the data (zero mean)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_variance

        # Step 6: Choose number of components if not specified
        if self.n_components is None:
            cumulative_variance = np.cumsum(self.explained_variance_ratio)
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            print(f"Automatically selected {self.n_components} components to retain {self.variance_threshold * 100}% variance.")

        # Step 7: Store the selected eigenvectors
        self.eigenvectors = eigenvectors[:, :self.n_components]

    @override
    def transform(self, X):
        """
        Project the data onto the principal components.
        :param X: Input data of shape (samples, features)
        :return: Transformed data of shape (samples, n_components)
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.eigenvectors)
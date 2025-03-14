import numpy as np
from matplotlib import pyplot as plt

from light import PCA

if __name__ == "__main__":
    # Generate a synthetic dataset
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples, 10 features

    # Apply PCA with automatic component selection
    pca = PCA(n_components=None, variance_threshold=0.7)
    X_pca = pca.fit_transform(X)

    print(f"Reduced data shape: {X_pca.shape}")

    # Plot the first 2 principal components
    if X_pca.shape[1] >= 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color='blue')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA using NumPy (Automatic Component Selection)')
        plt.show()

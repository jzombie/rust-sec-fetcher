from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# TODO: Add proper types
def plot_pca_explanation(
    concept_unit_embeddings, variance_threshold: float = 0.95
) -> int:
    # embeddings: np.ndarray of shape (N, 1024)
    pca = PCA(svd_solver="full")
    pca.fit(concept_unit_embeddings)

    explained = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(np.arange(1, len(explained) + 1), explained)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.grid(True)
    plt.axhline(variance_threshold, color="red", linestyle="--")
    plt.title("Explained Variance vs PCA Components")
    plt.show()

    return np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1

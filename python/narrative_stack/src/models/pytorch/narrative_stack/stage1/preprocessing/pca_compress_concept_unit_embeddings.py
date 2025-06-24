import logging
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple


def pca_compress_concept_unit_embeddings(
    concept_unit_embeddings: np.ndarray,
    n_components: Optional[int] = 200,
    pca: Optional[PCA] = None,
    stable: bool = False,
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA compression to concept-unit embeddings.

     Note:
        IncrementalPCA was considered for memory efficiency, but it introduces
        nondeterminism due to sensitivity to batch order and floating-point
        rounding. For reproducibility and cross-run consistency, full PCA with
        exact SVD is used instead.

    Parameters:
        concept_unit_embeddings (np.ndarray):
            The input embeddings array of shape (N, D), usually float32.
        n_components (int, optional):
            Number of PCA components to retain. Ignored if `pca` is provided.
        pca (PCA, optional):
            An existing fitted PCA model. If provided, `n_components` is ignored.
        stable (bool, default=False):
            If True, casts input to float64 before fitting/transforming
            for improved determinism and cosine preservation.

    Returns:
        Tuple[np.ndarray, PCA]:
            - Compressed embeddings of shape (N, n_components) (float32)
            - The fitted or reused PCA instance
    """
    if n_components is not None and pca is not None:
        logging.warning("Using existing PCA; n_components will be ignored.")

    X = (
        concept_unit_embeddings.astype(np.float64)
        if stable
        else concept_unit_embeddings
    )

    if pca is None:
        """
        Notes: To further ensure determinism:
            - Set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, etc.
            - Fix the random seed (not needed for "full", but good practice).
            - Use the same Python and sklearn/BLAS build across runs.
        """
        pca = PCA(
            n_components=n_components,
            svd_solver="full",  # Exact SVD (deterministic on same system, but not cross-platform bitwise identical)
        )
        compressed = pca.fit_transform(X)
    else:
        compressed = pca.transform(X)

    return compressed.astype(np.float32), pca

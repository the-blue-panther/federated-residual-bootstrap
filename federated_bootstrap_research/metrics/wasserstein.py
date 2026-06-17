"""Wasserstein distance metric for comparing bootstrap distributions."""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Optional


def compute_wasserstein(
    dist1: np.ndarray,
    dist2: np.ndarray,
) -> np.ndarray:
    """
    Compute Wasserstein distance between two distributions.
    
    Parameters
    ----------
    dist1 : np.ndarray
        First distribution (n_samples, p) or (n_samples,).
    dist2 : np.ndarray
        Second distribution (n_samples, p) or (n_samples,).
    
    Returns
    -------
    np.ndarray
        Wasserstein distance for each coefficient (p,) or scalar if 1D.
    """
    if dist1.ndim == 1:
        # Single variable
        return wasserstein_distance(dist1, dist2)
    else:
        # Multiple variables
        p = dist1.shape[1]
        distances = np.zeros(p)
        for j in range(p):
            distances[j] = wasserstein_distance(dist1[:, j], dist2[:, j])
        return distances

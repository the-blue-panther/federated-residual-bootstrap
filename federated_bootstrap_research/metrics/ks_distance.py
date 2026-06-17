"""Kolmogorov-Smirnov distance metric for comparing bootstrap distributions."""

import numpy as np
from scipy.stats import ks_2samp
from typing import Optional


def compute_ks_distance(
    dist1: np.ndarray,
    dist2: np.ndarray,
) -> np.ndarray:
    """
    Compute Kolmogorov-Smirnov distance between two distributions.
    
    Parameters
    ----------
    dist1 : np.ndarray
        First distribution (n_samples, p) or (n_samples,).
    dist2 : np.ndarray
        Second distribution (n_samples, p) or (n_samples,).
    
    Returns
    -------
    np.ndarray
        KS distance for each coefficient (p,) or scalar if 1D.
    """
    if dist1.ndim == 1:
        # Single variable
        result = ks_2samp(dist1, dist2)
        return result.statistic
    else:
        # Multiple variables
        p = dist1.shape[1]
        distances = np.zeros(p)
        for j in range(p):
            result = ks_2samp(dist1[:, j], dist2[:, j])
            distances[j] = result.statistic
        return distances

"""Bias computation."""

import numpy as np
from typing import Optional


def compute_bias(
    estimates: np.ndarray,
    true_beta: np.ndarray,
) -> np.ndarray:
    """
    Compute bias for each coefficient.
    
    Bias = E[hat_beta] - beta
    
    Parameters
    ----------
    estimates : np.ndarray
        Coefficient estimates (shape: (n_sim, p)).
    true_beta : np.ndarray
        True coefficient values (shape: (p,)).
    
    Returns
    -------
    np.ndarray
        Bias for each coefficient (shape: (p,)).
    """
    mean_estimates = np.mean(estimates, axis=0)
    return mean_estimates - true_beta

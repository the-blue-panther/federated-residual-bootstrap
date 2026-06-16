"""Mean Squared Error computation."""

import numpy as np
from typing import Optional


def compute_mse(
    estimates: np.ndarray,
    true_beta: np.ndarray,
) -> np.ndarray:
    """
    Compute Mean Squared Error for each coefficient.
    
    MSE = E[(hat_beta - beta)^2]
    
    Parameters
    ----------
    estimates : np.ndarray
        Coefficient estimates (shape: (n_sim, p)).
    true_beta : np.ndarray
        True coefficient values (shape: (p,)).
    
    Returns
    -------
    np.ndarray
        MSE for each coefficient (shape: (p,)).
    """
    squared_errors = (estimates - true_beta) ** 2
    return np.mean(squared_errors, axis=0)

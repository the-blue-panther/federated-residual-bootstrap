"""Theoretical standard error computation."""

import numpy as np
from typing import Optional


def compute_theoretical_se(
    X: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Compute theoretical standard errors.
    
    Var(hat_beta) = sigma^2 * (X^T X)^(-1)
    SE_j = sqrt(Var(hat_beta)_jj)
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (shape: (n, p)).
    sigma : float
        Error standard deviation.
    
    Returns
    -------
    np.ndarray
        Theoretical standard errors for each coefficient (shape: (p,)).
    """
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    var_cov = sigma ** 2 * XTX_inv
    return np.sqrt(np.diag(var_cov))

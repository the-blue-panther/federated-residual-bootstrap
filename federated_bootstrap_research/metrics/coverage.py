"""Coverage probability computation."""

import numpy as np
from typing import Optional


def compute_coverage(
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    true_beta: np.ndarray,
) -> np.ndarray:
    """
    Compute coverage probability for each coefficient.
    
    Parameters
    ----------
    ci_lower : np.ndarray
        Lower bounds of confidence intervals (shape: (p,) or (n_sim, p)).
    ci_upper : np.ndarray
        Upper bounds of confidence intervals (shape: (p,) or (n_sim, p)).
    true_beta : np.ndarray
        True coefficient values (shape: (p,)).
    
    Returns
    -------
    np.ndarray
        Coverage probability for each coefficient (shape: (p,)).
    """
    if ci_lower.ndim == 1:
        # Single confidence interval
        covered = (ci_lower <= true_beta) & (true_beta <= ci_upper)
        return covered.astype(float)
    else:
        # Multiple simulations
        n_sim, p = ci_lower.shape
        covered = np.zeros(p)
        for j in range(p):
            covered[j] = np.mean(
                (ci_lower[:, j] <= true_beta[j]) & (true_beta[j] <= ci_upper[:, j])
            )
        return covered

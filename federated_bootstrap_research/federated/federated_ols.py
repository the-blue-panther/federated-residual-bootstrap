"""Federated Ordinary Least Squares implementation.

Implements exact federated OLS where each site computes local sufficient
statistics (X^T X and X^T y) and the server aggregates them.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class FederatedOLS:
    """
    Federated Ordinary Least Squares estimator.
    
    Each site computes:
        XTX_m = X_m^T X_m
        XTy_m = X_m^T y_m
    
    Server aggregates:
        XTX_global = sum(XTX_m)
        XTy_global = sum(XTy_m)
    
    Then solves:
        beta_fed = solve(XTX_global, XTy_global)
    
    This is numerically identical to centralized OLS.
    """
    
    def __init__(self):
        """Initialize the federated OLS estimator."""
        self.beta_hat = None
        self.XTX_global = None
        self.XTy_global = None
        self.site_stats = []
    
    def compute_site_stats(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute local sufficient statistics for a site.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix for this site (n_m, p).
        y : np.ndarray
            Response vector for this site (n_m,).
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'XTX' and 'XTy' for this site.
        """
        XTX = X.T @ X
        XTy = X.T @ y
        return {
            "XTX": XTX,
            "XTy": XTy,
            "n": X.shape[0],
        }
    
    def aggregate_stats(
        self,
        site_stats: List[Dict[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate local sufficient statistics from all sites.
        
        Parameters
        ----------
        site_stats : List[Dict[str, np.ndarray]]
            List of dictionaries from each site containing 'XTX' and 'XTy'.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Global XTX and XTy matrices.
        """
        if not site_stats:
            raise ValueError("No site statistics provided")
        
        p = site_stats[0]["XTX"].shape[0]
        XTX_global = np.zeros((p, p))
        XTy_global = np.zeros(p)
        
        for stats in site_stats:
            XTX_global += stats["XTX"]
            XTy_global += stats["XTy"]
        
        self.XTX_global = XTX_global
        self.XTy_global = XTy_global
        self.site_stats = site_stats
        
        return XTX_global, XTy_global
    
    def fit(
        self,
        partitions: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Fit federated OLS using partitioned data.
        
        Parameters
        ----------
        partitions : List[Dict[str, np.ndarray]]
            List of dictionaries, each containing 'X' and 'y' for a site.
        
        Returns
        -------
        np.ndarray
            Federated OLS coefficient estimates.
        """
        # Compute local statistics
        site_stats = []
        for partition in partitions:
            X = partition["X"]
            y = partition["y"]
            stats = self.compute_site_stats(X, y)
            site_stats.append(stats)
        
        # Aggregate
        XTX_global, XTy_global = self.aggregate_stats(site_stats)
        
        # Solve
        self.beta_hat = np.linalg.solve(XTX_global, XTy_global)
        
        return self.beta_hat
    
    def fit_from_stats(
        self,
        XTX_global: np.ndarray,
        XTy_global: np.ndarray,
    ) -> np.ndarray:
        """
        Fit federated OLS from pre-aggregated statistics.
        
        Parameters
        ----------
        XTX_global : np.ndarray
            Global X^T X matrix.
        XTy_global : np.ndarray
            Global X^T y vector.
        
        Returns
        -------
        np.ndarray
            Federated OLS coefficient estimates.
        """
        self.XTX_global = XTX_global
        self.XTy_global = XTy_global
        self.beta_hat = np.linalg.solve(XTX_global, XTy_global)
        return self.beta_hat
    
    def get_site_stats(self) -> List[Dict[str, np.ndarray]]:
        """Return the local site statistics."""
        return self.site_stats


def centralized_ols(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Centralized OLS for comparison.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n, p).
    y : np.ndarray
        Response vector (n,).
    
    Returns
    -------
    np.ndarray
        OLS coefficient estimates.
    """
    XTX = X.T @ X
    XTy = X.T @ y
    return np.linalg.solve(XTX, XTy)

"""Centralized residual bootstrap implementation.

This module implements the classical centralized residual bootstrap for
ordinary least squares regression. This serves as the gold-standard benchmark
for future federated bootstrap methods.
"""

import numpy as np
from typing import Optional, Dict, Any


class CentralizedResidualBootstrap:
    """
    Centralized residual bootstrap for OLS regression.
    
    This class implements the classical residual bootstrap procedure:
    1. Fit OLS to get beta_hat and residuals
    2. Generate bootstrap samples by resampling residuals
    3. Refit OLS on each bootstrap sample
    4. Compute bootstrap standard errors and confidence intervals
    
    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (B). Default: 1000.
    confidence_level : float
        Confidence level for percentile intervals. Must be between 0 and 1.
        Default: 0.95.
    random_state : Optional[int]
        Random seed for reproducibility. Default: None.
    
    Attributes
    ----------
    beta_hat : np.ndarray
        OLS coefficient estimates from original data.
    residuals : np.ndarray
        Residuals from original fit.
    y_hat : np.ndarray
        Fitted values from original fit.
    bootstrap_betas : np.ndarray
        Bootstrap coefficient estimates (B x p).
    bootstrap_se : np.ndarray
        Bootstrap standard errors (p,).
    ci_lower : np.ndarray
        Lower confidence interval bounds (p,).
    ci_upper : np.ndarray
        Upper confidence interval bounds (p,).
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        """Initialize the centralized residual bootstrap."""
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Attributes to be set during fit
        self.beta_hat = None
        self.residuals = None
        self.y_hat = None
        self.bootstrap_betas = None
        self.bootstrap_se = None
        self.ci_lower = None
        self.ci_upper = None
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit the centralized residual bootstrap model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n, p).
        y : np.ndarray
            Response vector of shape (n,).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'beta_hat': OLS coefficient estimates
            - 'bootstrap_betas': Bootstrap coefficient matrix (B x p)
            - 'bootstrap_se': Bootstrap standard errors
            - 'ci_lower': Lower confidence interval bounds
            - 'ci_upper': Upper confidence interval bounds
        """
        self.X = X
        self.y = y
        
        # Step 1: Fit OLS on original data
        self._fit_ols()
        
        # Step 2: Compute residuals and fitted values
        self._compute_residuals()
        
        # Step 3: Generate bootstrap samples and refit
        self._bootstrap_loop()
        
        # Step 4: Compute standard errors
        self._compute_standard_errors()
        
        # Step 5: Compute confidence intervals
        self._compute_confidence_intervals()
        
        return {
            "beta_hat": self.beta_hat,
            "bootstrap_betas": self.bootstrap_betas,
            "bootstrap_se": self.bootstrap_se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }
    
    def _fit_ols(self) -> None:
        """
        Fit OLS using the normal equations.
        
        Computes: beta_hat = (X'X)^(-1) X'y
        """
        XTX = self.X.T @ self.X
        XTy = self.X.T @ self.y
        self.beta_hat = np.linalg.inv(XTX) @ XTy
    
    def _compute_residuals(self) -> None:
        """Compute residuals and fitted values from OLS fit."""
        self.y_hat = self.X @ self.beta_hat
        self.residuals = self.y - self.y_hat
    
    def _bootstrap_loop(self) -> None:
        """
        Perform bootstrap iterations.
        
        For each iteration:
        1. Sample residuals with replacement
        2. Generate y_star = y_hat + e_star
        3. Refit OLS to get beta_star
        """
        n = self.X.shape[0]
        p = self.X.shape[1]
        
        self.bootstrap_betas = np.zeros((self.n_bootstrap, p))
        
        for i in range(self.n_bootstrap):
            # Sample residuals with replacement
            e_star = self.rng.choice(self.residuals, size=n, replace=True)
            
            # Generate bootstrap response
            y_star = self.y_hat + e_star
            
            # Refit OLS
            XTX = self.X.T @ self.X
            XTy_star = self.X.T @ y_star
            beta_star = np.linalg.inv(XTX) @ XTy_star
            
            self.bootstrap_betas[i, :] = beta_star
    
    def _compute_standard_errors(self) -> None:
        """
        Compute bootstrap standard errors.
        
        SE_j = standard deviation of beta*_j across bootstrap samples
        """
        self.bootstrap_se = np.std(self.bootstrap_betas, axis=0, ddof=1)
    
    def _compute_confidence_intervals(self) -> None:
        """
        Compute percentile confidence intervals.
        
        For confidence level 1-alpha, compute:
        - Lower bound: alpha/2 percentile
        - Upper bound: 1-alpha/2 percentile
        """
        alpha = 1.0 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1.0 - alpha / 2) * 100
        
        self.ci_lower = np.percentile(
            self.bootstrap_betas,
            lower_percentile,
            axis=0,
            keepdims=False
        )
        self.ci_upper = np.percentile(
            self.bootstrap_betas,
            upper_percentile,
            axis=0,
            keepdims=False
        )

"""Heteroscedastic error data generator.

Generates synthetic data with variance depending on covariates.
"""

import numpy as np
from typing import Optional, Tuple


class HeteroscedasticDataGenerator:
    """
    Generates synthetic data with heteroscedastic errors.
    
    Variance depends on covariates: sigma_i^2 = 1 + x_{i1}^2
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    beta : np.ndarray
        Regression coefficients.
    sigma : float
        Base error standard deviation.
    random_state : Optional[int]
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float,
        random_state: Optional[int] = None,
    ):
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.default_rng(random_state)
    
    def generate_features(self) -> np.ndarray:
        """Generate feature matrix X with independent standard normal entries."""
        return self.rng.normal(loc=0.0, scale=1.0, size=(self.n, self.p))
    
    def generate_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Generate heteroscedastic errors.
        
        Variance depends on first covariate: sigma_i = sigma * sqrt(1 + x_{i1}^2)
        """
        # Base errors
        z = self.rng.normal(loc=0.0, scale=1.0, size=self.n)
        
        # Variance depends on first covariate
        sigma_i = self.sigma * np.sqrt(1.0 + X[:, 0] ** 2)
        
        # Scale errors
        errors = z * sigma_i
        return errors
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the complete dataset (X, y) with heteroscedastic errors.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X and response vector y.
        """
        X = self.generate_features()
        epsilon = self.generate_errors(X)
        y = X @ self.beta + epsilon
        return X, y

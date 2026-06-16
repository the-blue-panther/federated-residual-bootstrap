"""Skewed error data generator.

Generates synthetic data with skewed errors using exponential distribution.
"""

import numpy as np
from typing import Optional, Tuple


class SkewedDataGenerator:
    """
    Generates synthetic data with skewed errors.
    
    Uses Exp(1) - 1 scaled to match sigma^2 variance.
    Exp(1) has variance 1, so scaling factor is sigma.
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    beta : np.ndarray
        Regression coefficients.
    sigma : float
        Target error standard deviation.
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
    
    def generate_errors(self) -> np.ndarray:
        """
        Generate skewed errors from Exp(1) - 1 scaled to match sigma^2 variance.
        
        Exp(1) has mean 1, variance 1.
        Exp(1) - 1 has mean 0, variance 1.
        """
        errors = self.rng.exponential(scale=1.0, size=self.n) - 1.0
        # Scale to match target sigma
        errors = errors * self.sigma
        return errors
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the complete dataset (X, y).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X and response vector y.
        """
        X = self.generate_features()
        epsilon = self.generate_errors()
        y = X @ self.beta + epsilon
        return X, y

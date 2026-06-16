"""Heavy-tailed error data generator.

Generates synthetic data with t-distributed errors.
"""

import numpy as np
from typing import Optional, Tuple


class HeavyTailedDataGenerator:
    """
    Generates synthetic data with heavy-tailed errors.
    
    Uses t_3 distribution scaled to match sigma^2 variance.
    
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
        Generate t_3 errors scaled to match sigma^2 variance.
        
        t_3 has variance 3, so we scale by sigma/sqrt(3).
        """
        errors = self.rng.standard_t(df=3, size=self.n)
        # Scale to match target sigma
        errors = errors * (self.sigma / np.sqrt(3))
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

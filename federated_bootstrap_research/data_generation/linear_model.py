"""Linear model data generator for federated bootstrap simulations."""

import numpy as np
from typing import Optional, Tuple


class LinearModelDataGenerator:
    """
    Generates synthetic data from a linear model: Y = X * beta + epsilon.
    
    The generator simulates data according to:
        Y = X * beta + epsilon, where epsilon ~ N(0, sigma^2)
    
    Parameters
    ----------
    n : int
        Sample size (number of observations).
    p : int
        Number of features (dimension of X).
    beta : np.ndarray
        Regression coefficients (shape: p,).
    sigma : float
        Standard deviation of the error term.
    random_state : Optional[int]
        Seed for reproducible random number generation.
    
    Attributes
    ----------
    rng : np.random.Generator
        Random number generator instance.
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float,
        random_state: Optional[int] = None,
    ):
        """Initialize the linear model data generator."""
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.rng = np.random.default_rng(random_state)
        
    def generate_features(self) -> np.ndarray:
        """
        Generate feature matrix X with independent standard normal entries.
        
        Returns
        -------
        np.ndarray
            Feature matrix of shape (n, p).
        """
        return self.rng.normal(loc=0.0, scale=1.0, size=(self.n, self.p))
    
    def generate_errors(self) -> np.ndarray:
        """
        Generate error terms from a normal distribution.
        
        Returns
        -------
        np.ndarray
            Error vector of shape (n,).
        """
        return self.rng.normal(loc=0.0, scale=self.sigma, size=self.n)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the complete dataset (X, y).
        
        The response is computed as:
            y = X @ beta + epsilon
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X (shape: n x p) and response vector y (shape: n,).
        """
        X = self.generate_features()
        epsilon = self.generate_errors()
        y = X @ self.beta + epsilon
        return X, y

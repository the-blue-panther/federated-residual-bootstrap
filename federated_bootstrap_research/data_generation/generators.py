"""Factory functions for creating and using data generators."""

from typing import Optional, Tuple
import numpy as np
from federated_bootstrap_research.data_generation.linear_model import (
    LinearModelDataGenerator,
)


def create_generator(
    n: int,
    p: int,
    beta: np.ndarray,
    sigma: float,
    random_state: Optional[int] = None,
) -> LinearModelDataGenerator:
    """
    Create a LinearModelDataGenerator instance with the given parameters.
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    beta : np.ndarray
        Regression coefficients.
    sigma : float
        Error standard deviation.
    random_state : Optional[int]
        Random seed for reproducibility.
    
    Returns
    -------
    LinearModelDataGenerator
        Configured generator instance.
    """
    return LinearModelDataGenerator(
        n=n,
        p=p,
        beta=beta,
        sigma=sigma,
        random_state=random_state,
    )


def generate_dataset(
    n: int,
    p: int,
    beta: np.ndarray,
    sigma: float,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate a dataset in one call.
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    beta : np.ndarray
        Regression coefficients.
    sigma : float
        Error standard deviation.
    random_state : Optional[int]
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X and response vector y.
    """
    generator = create_generator(
        n=n,
        p=p,
        beta=beta,
        sigma=sigma,
        random_state=random_state,
    )
    return generator.generate()

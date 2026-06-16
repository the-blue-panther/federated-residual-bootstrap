"""Factory functions for creating and using data generators."""

from typing import Optional, Tuple
import numpy as np
from federated_bootstrap_research.data_generation.linear_model import (
    LinearModelDataGenerator,
)
from federated_bootstrap_research.data_generation.heavy_tailed import (
    HeavyTailedDataGenerator,
)
from federated_bootstrap_research.data_generation.skewed import (
    SkewedDataGenerator,
)
from federated_bootstrap_research.data_generation.heteroscedastic import (
    HeteroscedasticDataGenerator,
)


def create_generator(
    n: int,
    p: int,
    beta: np.ndarray,
    sigma: float,
    random_state: Optional[int] = None,
    distribution: str = "iid",
):
    """
    Create a data generator with the specified distribution.
    
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
    distribution : str
        Error distribution type. Options: "iid", "heavy_tailed", "skewed", "heteroscedastic"
    
    Returns
    -------
    Generator instance
    """
    if distribution == "heavy_tailed":
        return HeavyTailedDataGenerator(
            n=n,
            p=p,
            beta=beta,
            sigma=sigma,
            random_state=random_state,
        )
    elif distribution == "skewed":
        return SkewedDataGenerator(
            n=n,
            p=p,
            beta=beta,
            sigma=sigma,
            random_state=random_state,
        )
    elif distribution == "heteroscedastic":
        return HeteroscedasticDataGenerator(
            n=n,
            p=p,
            beta=beta,
            sigma=sigma,
            random_state=random_state,
        )
    else:  # "iid" default
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
    distribution: str = "iid",
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
    distribution : str
        Error distribution type. Options: "iid", "heavy_tailed", "skewed", "heteroscedastic"
    
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
        distribution=distribution,
    )
    return generator.generate()

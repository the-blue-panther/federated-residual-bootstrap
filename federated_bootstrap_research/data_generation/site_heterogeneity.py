"""Cross-site heterogeneity generators for federated bootstrap validation.

Implements various scenarios where different sites have different data-generating mechanisms.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple


class VarianceHeterogeneityGenerator:
    """
    Variance heterogeneity: Different sites have different error variances.
    
    Scenario A:
    - Site 1: epsilon ~ N(0, 1)
    - Site 2: epsilon ~ N(0, 4)
    - Site 3: epsilon ~ N(0, 9)
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float = 1.5,
        random_state: Optional[int] = None,
        num_sites: int = 3,
    ):
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.num_sites = num_sites
        self.rng = np.random.default_rng(random_state)
        
        # Variance multipliers for each site
        self.variance_multipliers = [1.0, 4.0, 9.0]
        # Standard deviation for each site
        self.site_sigmas = [sigma * np.sqrt(v) for v in self.variance_multipliers]
    
    def generate(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate heterogeneous data.
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of site data with 'X' and 'y'.
        """
        n_per_site = self.n // self.num_sites
        partitions = []
        
        for site_idx in range(self.num_sites):
            # Generate features
            X = self.rng.normal(0, 1, size=(n_per_site, self.p))
            
            # Generate errors with site-specific variance
            epsilon = self.rng.normal(0, self.site_sigmas[site_idx], size=n_per_site)
            
            # Generate response
            y = X @ self.beta + epsilon
            
            partitions.append({"X": X, "y": y})
        
        return partitions


class FeatureHeterogeneityGenerator:
    """
    Feature distribution heterogeneity: Different sites have different X means.
    
    Scenario B:
    - Site 1: X ~ N(0, 1)
    - Site 2: X ~ N(2, 1)
    - Site 3: X ~ N(-2, 1)
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float = 1.5,
        random_state: Optional[int] = None,
        num_sites: int = 3,
    ):
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.num_sites = num_sites
        self.rng = np.random.default_rng(random_state)
        
        # Mean shifts for each site
        self.mean_shifts = [0.0, 2.0, -2.0]
    
    def generate(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate heterogeneous data.
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of site data with 'X' and 'y'.
        """
        n_per_site = self.n // self.num_sites
        partitions = []
        
        for site_idx in range(self.num_sites):
            # Generate features with site-specific mean shift
            X = self.rng.normal(
                self.mean_shifts[site_idx], 
                1.0, 
                size=(n_per_site, self.p)
            )
            
            # Same noise distribution
            epsilon = self.rng.normal(0, self.sigma, size=n_per_site)
            
            # Generate response
            y = X @ self.beta + epsilon
            
            partitions.append({"X": X, "y": y})
        
        return partitions


class ResidualShapeHeterogeneityGenerator:
    """
    Residual shape heterogeneity: Different sites have different error distributions.
    
    Scenario C:
    - Site 1: Gaussian errors
    - Site 2: Heavy-tailed t(3) errors
    - Site 3: Skewed Exponential errors
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float = 1.5,
        random_state: Optional[int] = None,
        num_sites: int = 3,
    ):
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.num_sites = num_sites
        self.rng = np.random.default_rng(random_state)
    
    def generate(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate heterogeneous data.
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of site data with 'X' and 'y'.
        """
        n_per_site = self.n // self.num_sites
        partitions = []
        
        for site_idx in range(self.num_sites):
            # Generate features
            X = self.rng.normal(0, 1, size=(n_per_site, self.p))
            
            # Generate errors with different distributions
            if site_idx == 0:
                # Gaussian
                epsilon = self.rng.normal(0, self.sigma, size=n_per_site)
            elif site_idx == 1:
                # Heavy-tailed t(3)
                epsilon = self.rng.standard_t(3, size=n_per_site) * (self.sigma / np.sqrt(3))
            else:
                # Skewed Exponential
                epsilon = (self.rng.exponential(1.0, size=n_per_site) - 1.0) * self.sigma
            
            # Generate response
            y = X @ self.beta + epsilon
            
            partitions.append({"X": X, "y": y})
        
        return partitions


class CombinedHeterogeneityGenerator:
    """
    Combined heterogeneity: Different sites differ in variance, shape, and feature means.
    
    Scenario D:
    - Site 1: Gaussian, small variance, X ~ N(0, 1)
    - Site 2: Heavy-tailed, medium variance, X ~ N(2, 1)
    - Site 3: Skewed, large variance, X ~ N(-2, 1)
    """
    
    def __init__(
        self,
        n: int,
        p: int,
        beta: np.ndarray,
        sigma: float = 1.5,
        random_state: Optional[int] = None,
        num_sites: int = 3,
    ):
        self.n = n
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.num_sites = num_sites
        self.rng = np.random.default_rng(random_state)
        
        # Site configurations
        self.site_configs = [
            {
                "mean_shift": 0.0,
                "variance_mult": 1.0,
                "distribution": "gaussian"
            },
            {
                "mean_shift": 2.0,
                "variance_mult": 4.0,
                "distribution": "heavy_tailed"
            },
            {
                "mean_shift": -2.0,
                "variance_mult": 9.0,
                "distribution": "skewed"
            }
        ]
    
    def generate(self) -> List[Dict[str, np.ndarray]]:
        """
        Generate heterogeneous data.
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of site data with 'X' and 'y'.
        """
        n_per_site = self.n // self.num_sites
        partitions = []
        
        for site_idx, config in enumerate(self.site_configs):
            # Generate features with site-specific mean shift
            X = self.rng.normal(
                config["mean_shift"],
                1.0,
                size=(n_per_site, self.p)
            )
            
            # Generate errors with different distributions and variances
            site_sigma = self.sigma * np.sqrt(config["variance_mult"])
            
            if config["distribution"] == "gaussian":
                epsilon = self.rng.normal(0, site_sigma, size=n_per_site)
            elif config["distribution"] == "heavy_tailed":
                epsilon = self.rng.standard_t(3, size=n_per_site) * (site_sigma / np.sqrt(3))
            else:  # skewed
                epsilon = (self.rng.exponential(1.0, size=n_per_site) - 1.0) * site_sigma
            
            # Generate response
            y = X @ self.beta + epsilon
            
            partitions.append({"X": X, "y": y})
        
        return partitions


def create_heterogeneous_data(
    n: int,
    p: int,
    beta: np.ndarray,
    sigma: float = 1.5,
    random_state: Optional[int] = None,
    num_sites: int = 3,
    heterogeneity_type: str = "variance",
) -> List[Dict[str, np.ndarray]]:
    """
    Factory function for creating heterogeneous data.
    
    Parameters
    ----------
    n : int
        Total sample size.
    p : int
        Number of features.
    beta : np.ndarray
        Regression coefficients.
    sigma : float
        Error standard deviation.
    random_state : Optional[int]
        Random seed.
    num_sites : int
        Number of federated sites.
    heterogeneity_type : str
        Type of heterogeneity: "variance", "feature", "residual_shape", "combined"
    
    Returns
    -------
    List[Dict[str, np.ndarray]]
        List of site data with 'X' and 'y'.
    """
    if heterogeneity_type == "variance":
        generator = VarianceHeterogeneityGenerator(
            n=n, p=p, beta=beta, sigma=sigma,
            random_state=random_state, num_sites=num_sites
        )
    elif heterogeneity_type == "feature":
        generator = FeatureHeterogeneityGenerator(
            n=n, p=p, beta=beta, sigma=sigma,
            random_state=random_state, num_sites=num_sites
        )
    elif heterogeneity_type == "residual_shape":
        generator = ResidualShapeHeterogeneityGenerator(
            n=n, p=p, beta=beta, sigma=sigma,
            random_state=random_state, num_sites=num_sites
        )
    elif heterogeneity_type == "combined":
        generator = CombinedHeterogeneityGenerator(
            n=n, p=p, beta=beta, sigma=sigma,
            random_state=random_state, num_sites=num_sites
        )
    else:
        raise ValueError(f"Unknown heterogeneity type: {heterogeneity_type}")
    
    return generator.generate()

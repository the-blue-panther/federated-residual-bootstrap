"""Local residual bootstrap implementation.

Each site resamples only from its own residual pool.
This is the simplest federated bootstrap method.
"""

import numpy as np
from typing import Optional, Dict, Any, List

from federated_bootstrap_research.federated.federated_ols import FederatedOLS


class LocalResidualBootstrap:
    """
    Local residual bootstrap for federated learning.
    
    Each site computes residuals locally and resamples only from its own
    residual pool. This is the simplest federated bootstrap method.
    
    Algorithm:
    1. Compute federated OLS to get beta_fed
    2. For each site, compute local residuals and center them
    3. For each bootstrap iteration:
       a. Each site resamples from its centered residuals
       b. Generate y_star using beta_fed and resampled residuals
       c. Run federated OLS on the bootstrap sample
       d. Store beta_fed_star
    4. Compute standard errors and confidence intervals
    
    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (B). Default: 1000.
    confidence_level : float
        Confidence level for percentile intervals. Default: 0.95.
    random_state : Optional[int]
        Random seed for reproducibility. Default: None.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ):
        """Initialize the local residual bootstrap."""
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # Attributes to be set during fit
        self.beta_hat = None
        self.bootstrap_betas = None
        self.bootstrap_se = None
        self.ci_lower = None
        self.ci_upper = None
        self.partitions = None
        self.site_centered_residuals = []
        self.site_y_hat = []
    
    def fit(
        self,
        partitions: List[Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Fit the local residual bootstrap model.
        
        Parameters
        ----------
        partitions : List[Dict[str, np.ndarray]]
            List of dictionaries, each containing 'X' and 'y' for a site.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'beta_hat': Federated OLS coefficient estimates
            - 'bootstrap_betas': Bootstrap coefficient matrix (B x p)
            - 'bootstrap_se': Bootstrap standard errors
            - 'ci_lower': Lower confidence interval bounds
            - 'ci_upper': Upper confidence interval bounds
        """
        self.partitions = partitions
        
        # Step 1: Fit federated OLS
        federated_ols = FederatedOLS()
        self.beta_hat = federated_ols.fit(partitions)
        
        # Step 2: Compute residuals for each site
        self._compute_site_residuals()
        
        # Step 3: Bootstrap loop
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
    
    def _compute_site_residuals(self) -> None:
        """
        Compute residuals for each site and center them.
        
        For each site:
            y_hat_m = X_m @ beta_fed
            e_m = y_m - y_hat_m
            e_m_centered = e_m - mean(e_m)
        """
        self.site_centered_residuals = []
        self.site_y_hat = []
        
        for partition in self.partitions:
            X = partition["X"]
            y = partition["y"]
            
            # Fitted values using federated beta
            y_hat = X @ self.beta_hat
            self.site_y_hat.append(y_hat)
            
            # Residuals and centering
            residuals = y - y_hat
            residual_mean = np.mean(residuals)
            centered_residuals = residuals - residual_mean
            self.site_centered_residuals.append(centered_residuals)
    
    def _bootstrap_loop(self) -> None:
        """
        Perform bootstrap iterations.
        
        For each iteration:
        1. Each site samples from its centered residuals
        2. Generate y_star for each site
        3. Run federated OLS on all bootstrap samples
        4. Store beta_fed_star
        """
        p = self.partitions[0]["X"].shape[1]
        self.bootstrap_betas = np.zeros((self.n_bootstrap, p))
        
        for b in range(self.n_bootstrap):
            # Generate bootstrap partitions
            bootstrap_partitions = []
            
            for site_idx, partition in enumerate(self.partitions):
                X = partition["X"]
                n_site = X.shape[0]
                
                # Sample residuals from this site's centered residual pool
                e_star = self.rng.choice(
                    self.site_centered_residuals[site_idx],
                    size=n_site,
                    replace=True,
                )
                
                # Generate bootstrap response
                y_star = self.site_y_hat[site_idx] + e_star
                
                bootstrap_partitions.append({
                    "X": X,
                    "y": y_star,
                })
            
            # Run federated OLS on bootstrap sample
            federated_ols = FederatedOLS()
            beta_star = federated_ols.fit(bootstrap_partitions)
            
            self.bootstrap_betas[b, :] = beta_star
    
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
            keepdims=False,
        )
        self.ci_upper = np.percentile(
            self.bootstrap_betas,
            upper_percentile,
            axis=0,
            keepdims=False,
        )

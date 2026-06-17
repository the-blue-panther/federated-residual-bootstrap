"""Convergence rate study for Local Residual Bootstrap.

Estimates the convergence rate of Wasserstein distance as n increases.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
from scipy import stats

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
    LocalResidualBootstrap,
)
from federated_bootstrap_research.federated.partition import FederatedPartitioner
from federated_bootstrap_research.metrics.wasserstein import compute_wasserstein


def run_convergence_rate_study(
    n_values: List[int] = [100, 250, 500, 1000, 2500, 5000, 10000],
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 200,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/phase_2_validation/convergence_rate",
) -> Dict[str, Any]:
    """
    Run convergence rate study.
    
    Estimates b where W_n = O(n^b) via linear regression on log-log scale.
    
    Parameters
    ----------
    n_values : List[int]
        Sample sizes to test.
    p : int
        Number of features.
    beta : Optional[np.ndarray]
        True coefficients.
    sigma : float
        Error standard deviation.
    num_sites : int
        Number of federated sites.
    n_bootstrap : int
        Number of bootstrap iterations.
    n_mc : int
        Number of Monte Carlo simulations per n.
    random_state : Optional[int]
        Random seed.
    distribution : str
        Error distribution type.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including convergence rate estimates.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("CONVERGENCE RATE STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Feature dimension (p):      {p}")
    print(f"  True beta:                  {beta}")
    print(f"  Error sigma:                {sigma}")
    print(f"  Number of sites:            {num_sites}")
    print(f"  Bootstrap iterations:       {n_bootstrap}")
    print(f"  Monte Carlo runs per n:     {n_mc}")
    print(f"  Distribution:               {distribution}")
    print(f"  Sample sizes:               {n_values}")
    print(f"  Random seed:                {random_state}")
    
    # Store Wasserstein distances by n
    wasserstein_means = []
    wasserstein_stds = []
    
    print("\n" + "-" * 70)
    print("Running convergence rate simulations...")
    print("-" * 70)
    
    for n_idx, n in enumerate(n_values, 1):
        print(f"\n  Processing n = {n} ({n_idx}/{len(n_values)})")
        
        wasserstein_all = []
        
        for i in range(n_mc):
            if (i + 1) % 50 == 0:
                print(f"    Completed {i + 1} / {n_mc} simulations")
            
            # Generate data
            sim_seed = (random_state + n_idx * 100000 + i * 1000) if random_state is not None else None
            X, y = generate_dataset(
                n=n,
                p=p,
                beta=beta,
                sigma=sigma,
                random_state=sim_seed,
                distribution=distribution,
            )
            
            # Centralized bootstrap
            central_bootstrap = CentralizedResidualBootstrap(
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_state=sim_seed,
            )
            central_results = central_bootstrap.fit(X, y)
            
            # Partition data
            partitioner = FederatedPartitioner(
                num_sites=num_sites,
                random_state=sim_seed,
            )
            partitions = partitioner.partition(X, y)
            
            # Local residual bootstrap
            local_bootstrap = LocalResidualBootstrap(
                n_bootstrap=n_bootstrap,
                confidence_level=0.95,
                random_state=sim_seed,
            )
            local_results = local_bootstrap.fit(partitions)
            
            # Wasserstein distance
            w_dist = compute_wasserstein(
                central_results["bootstrap_betas"],
                local_results["bootstrap_betas"],
            )
            wasserstein_all.append(np.mean(w_dist))
        
        # Store results
        wasserstein_mean = np.mean(wasserstein_all)
        wasserstein_std = np.std(wasserstein_all)
        wasserstein_means.append(wasserstein_mean)
        wasserstein_stds.append(wasserstein_std)
        
        print(f"    Mean Wasserstein: {wasserstein_mean:.6f} (std: {wasserstein_std:.6f})")
    
    # Fit convergence rate: log(W_n) = a + b * log(n)
    log_n = np.log(n_values)
    log_w = np.log(wasserstein_means)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_w)
    
    # Confidence interval for slope
    n_obs = len(n_values)
    t_value = stats.t.ppf(0.975, n_obs - 2)
    slope_ci_lower = slope - t_value * std_err
    slope_ci_upper = slope + t_value * std_err
    
    # R-squared
    r_squared = r_value ** 2
    
    print("\n" + "-" * 70)
    print("Convergence Rate Analysis")
    print("-" * 70)
    print(f"\n  Model: log(W_n) = a + b * log(n)")
    print(f"  Estimated b: {slope:.4f}")
    print(f"  95% CI for b: [{slope_ci_lower:.4f}, {slope_ci_upper:.4f}]")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  p-value: {p_value:.4e}")
    
    # Interpretation
    if slope < -0.3 and slope > -0.7:
        interpretation = "Slope suggests ~O(n^-1/2) convergence (plausible)"
    elif slope < -0.7:
        interpretation = "Slope suggests faster than O(n^-1/2) convergence"
    elif slope < -0.1:
        interpretation = "Slope suggests slower than O(n^-1/2) convergence"
    else:
        interpretation = "Slope suggests very slow or no convergence"
    
    print(f"\n  Interpretation: {interpretation}")
    
    # Check if Wasserstein appears to approach zero
    wasserstein_decreasing = all(
        wasserstein_means[i+1] < wasserstein_means[i]
        for i in range(len(wasserstein_means)-1)
    )
    
    print(f"\n  Wasserstein decreasing: {'PASSED' if wasserstein_decreasing else 'FAILED'}")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Summary table
        summary_df = pd.DataFrame({
            "n": n_values,
            "wasserstein_mean": wasserstein_means,
            "wasserstein_std": wasserstein_stds,
            "log_n": log_n,
            "log_w": log_w,
        })
        summary_path = results_path / f"convergence_{distribution}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
        
        # Regression results
        regression_df = pd.DataFrame({
            "metric": ["slope", "slope_ci_lower", "slope_ci_upper", "r_squared", "p_value", "n_obs"],
            "value": [slope, slope_ci_lower, slope_ci_upper, r_squared, p_value, n_obs],
        })
        regression_path = results_path / f"convergence_regression_{distribution}.csv"
        regression_df.to_csv(regression_path, index=False)
        print(f"  Regression saved to: {regression_path}")
    
    print("\n" + "=" * 70)
    
    return {
        "slope": slope,
        "slope_ci_lower": slope_ci_lower,
        "slope_ci_upper": slope_ci_upper,
        "r_squared": r_squared,
        "p_value": p_value,
        "interpretation": interpretation,
        "wasserstein_decreasing": wasserstein_decreasing,
        "wasserstein_means": wasserstein_means,
        "wasserstein_stds": wasserstein_stds,
    }


if __name__ == "__main__":
    run_convergence_rate_study(
        n_values=[100, 250, 500, 1000, 2500, 5000, 10000],
        p=5,
        n_bootstrap=500,
        n_mc=50,  # Lower for quick testing
        random_state=42,
    )

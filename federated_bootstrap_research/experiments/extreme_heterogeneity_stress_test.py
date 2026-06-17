"""Extreme heterogeneity stress test for Local Residual Bootstrap.

Tests the hardest possible federated environment with extreme differences between sites.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
    LocalResidualBootstrap,
)
from federated_bootstrap_research.metrics import (
    compute_coverage,
    compute_bias,
    compute_mse,
)
from federated_bootstrap_research.metrics.wasserstein import compute_wasserstein
from federated_bootstrap_research.metrics.ks_distance import compute_ks_distance


def generate_extreme_heterogeneous_data(
    n: int,
    p: int,
    beta: np.ndarray,
    sigma: float = 1.5,
    random_state: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate extreme heterogeneous data.
    
    Scenario:
    - Site 1: X ~ N(-5, 1), Gaussian noise, variance = 1
    - Site 2: X ~ N(0, 1), Heavy-tailed noise, variance = 9
    - Site 3: X ~ N(5, 1), Skewed noise, variance = 25
    
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
    
    Returns
    -------
    List[Dict[str, np.ndarray]]
        List of site data with 'X' and 'y'.
    """
    rng = np.random.default_rng(random_state)
    n_per_site = n // 3
    partitions = []
    
    # Site configurations
    site_configs = [
        {
            "mean_shift": -5.0,
            "sigma_mult": 1.0,
            "distribution": "gaussian"
        },
        {
            "mean_shift": 0.0,
            "sigma_mult": 3.0,
            "distribution": "heavy_tailed"
        },
        {
            "mean_shift": 5.0,
            "sigma_mult": 5.0,
            "distribution": "skewed"
        }
    ]
    
    for config in site_configs:
        # Generate features with extreme mean shift
        X = rng.normal(config["mean_shift"], 1.0, size=(n_per_site, p))
        
        # Generate errors with different distributions and extreme variances
        site_sigma = sigma * config["sigma_mult"]
        
        if config["distribution"] == "gaussian":
            epsilon = rng.normal(0, site_sigma, size=n_per_site)
        elif config["distribution"] == "heavy_tailed":
            epsilon = rng.standard_t(3, size=n_per_site) * (site_sigma / np.sqrt(3))
        else:  # skewed
            epsilon = (rng.exponential(1.0, size=n_per_site) - 1.0) * site_sigma
        
        # Generate response
        y = X @ beta + epsilon
        
        partitions.append({"X": X, "y": y})
    
    return partitions


def run_extreme_heterogeneity_stress_test(
    n_values: List[int] = [1000, 5000, 10000],
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 50,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    save_results: bool = True,
    results_dir: str = "results/phase_2_75_audit/extreme_stress",
) -> Dict[str, Any]:
    """
    Run extreme heterogeneity stress test.
    
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
    confidence_level : float
        Confidence level for intervals.
    random_state : Optional[int]
        Random seed.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including extreme stress test metrics.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("EXTREME HETEROGENEITY STRESS TEST")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Feature dimension (p):      {p}")
    print(f"  True beta:                  {beta}")
    print(f"  Error sigma:                {sigma}")
    print(f"  Number of sites:            {num_sites}")
    print(f"  Bootstrap iterations:       {n_bootstrap}")
    print(f"  Monte Carlo runs per n:     {n_mc}")
    print(f"  Confidence level:           {confidence_level}")
    print(f"  Sample sizes:               {n_values}")
    print(f"  Random seed:                {random_state}")
    
    print("\n" + "-" * 70)
    print("Extreme Scenario:")
    print("  Site 1: X ~ N(-5, 1), Gaussian noise, variance = 1")
    print("  Site 2: X ~ N(0, 1), Heavy-tailed noise, variance = 9")
    print("  Site 3: X ~ N(5, 1), Skewed noise, variance = 25")
    print("-" * 70)
    
    results_by_n = []
    
    for n_idx, n in enumerate(n_values, 1):
        print(f"\n  Processing n = {n} ({n_idx}/{len(n_values)})")
        
        # Store results for this n
        central_beta_hats = []
        local_beta_hats = []
        central_ci_lower = []
        central_ci_upper = []
        local_ci_lower = []
        local_ci_upper = []
        central_bootstrap_betas = []
        local_bootstrap_betas = []
        
        for i in range(n_mc):
            if (i + 1) % 20 == 0:
                print(f"    Completed {i + 1} / {n_mc} simulations")
            
            # Generate extreme heterogeneous data
            sim_seed = (random_state + n_idx * 10000 + i * 1000) if random_state is not None else None
            if sim_seed is not None:
                sim_seed = abs(sim_seed) % 100000
            
            partitions = generate_extreme_heterogeneous_data(
                n=n,
                p=p,
                beta=beta,
                sigma=sigma,
                random_state=sim_seed,
            )
            
            # Combine partitions for centralized analysis
            X_combined = np.vstack([p["X"] for p in partitions])
            y_combined = np.concatenate([p["y"] for p in partitions])
            
            # Centralized bootstrap
            central_bootstrap = CentralizedResidualBootstrap(
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=sim_seed,
            )
            central_results = central_bootstrap.fit(X_combined, y_combined)
            central_beta_hats.append(central_results["beta_hat"])
            central_ci_lower.append(central_results["ci_lower"])
            central_ci_upper.append(central_results["ci_upper"])
            central_bootstrap_betas.append(central_results["bootstrap_betas"])
            
            # Local residual bootstrap
            local_bootstrap = LocalResidualBootstrap(
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=sim_seed,
            )
            local_results = local_bootstrap.fit(partitions)
            local_beta_hats.append(local_results["beta_hat"])
            local_ci_lower.append(local_results["ci_lower"])
            local_ci_upper.append(local_results["ci_upper"])
            local_bootstrap_betas.append(local_results["bootstrap_betas"])
        
        # Convert to arrays
        central_beta_hats = np.array(central_beta_hats)
        local_beta_hats = np.array(local_beta_hats)
        central_ci_lower = np.array(central_ci_lower)
        central_ci_upper = np.array(central_ci_upper)
        local_ci_lower = np.array(local_ci_lower)
        local_ci_upper = np.array(local_ci_upper)
        central_bootstrap_betas = np.array(central_bootstrap_betas)
        local_bootstrap_betas = np.array(local_bootstrap_betas)
        
        # Compute metrics
        central_bias = compute_bias(central_beta_hats, beta)
        central_mse = compute_mse(central_beta_hats, beta)
        central_coverage = compute_coverage(central_ci_lower, central_ci_upper, beta)
        central_avg_coverage = np.mean(central_coverage)
        
        local_bias = compute_bias(local_beta_hats, beta)
        local_mse = compute_mse(local_beta_hats, beta)
        local_coverage = compute_coverage(local_ci_lower, local_ci_upper, beta)
        local_avg_coverage = np.mean(local_coverage)
        
        # Distribution distances
        wasserstein_dist = np.mean([
            compute_wasserstein(central_bootstrap_betas[j], local_bootstrap_betas[j])
            for j in range(n_mc)
        ], axis=0)
        
        ks_dist = np.mean([
            compute_ks_distance(central_bootstrap_betas[j], local_bootstrap_betas[j])
            for j in range(n_mc)
        ], axis=0)
        
        coverage_diff = local_avg_coverage - central_avg_coverage
        
        # Store results for this n
        n_results = {
            "n": n,
            "central_bias": central_bias,
            "central_mse": central_mse,
            "central_coverage": central_coverage,
            "central_avg_coverage": central_avg_coverage,
            "local_bias": local_bias,
            "local_mse": local_mse,
            "local_coverage": local_coverage,
            "local_avg_coverage": local_avg_coverage,
            "coverage_diff": coverage_diff,
            "wasserstein": wasserstein_dist,
            "ks_distance": ks_dist,
            "mean_wasserstein": np.mean(wasserstein_dist),
            "mean_ks": np.mean(ks_dist),
        }
        results_by_n.append(n_results)
        
        print(f"    Avg coverage (central): {central_avg_coverage:.4f}")
        print(f"    Avg coverage (local):   {local_avg_coverage:.4f}")
        print(f"    Coverage diff:          {coverage_diff:+.4f}")
        print(f"    Mean Wasserstein:       {n_results['mean_wasserstein']:.6f}")
        print(f"    Mean KS:                {n_results['mean_ks']:.6f}")
        
        # Save results
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            p_dim = len(beta)
            summary_df = pd.DataFrame({
                "coefficient": [f"beta_{j}" for j in range(p_dim)],
                "central_bias": central_bias,
                "central_mse": central_mse,
                "central_coverage": central_coverage,
                "local_bias": local_bias,
                "local_mse": local_mse,
                "local_coverage": local_coverage,
                "wasserstein": wasserstein_dist,
                "ks_distance": ks_dist,
            })
            summary_path = results_path / f"extreme_stress_n_{n}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTREME STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"\n{'n':<8} {'Central Cov':<12} {'Local Cov':<12} {'Diff':<10} {'Wasserstein':<12} {'KS':<10}")
    print("-" * 70)
    for results in results_by_n:
        print(f"{results['n']:<8} {results['central_avg_coverage']:.4f}     {results['local_avg_coverage']:.4f}     {results['coverage_diff']:+.4f}    {results['mean_wasserstein']:.6f}   {results['mean_ks']:.6f}")
    
    print("\n" + "=" * 70)
    
    return results_by_n


if __name__ == "__main__":
    run_extreme_heterogeneity_stress_test(
        n_values=[1000, 5000, 10000],
        p=5,
        n_bootstrap=500,
        n_mc=20,
        random_state=42,
    )

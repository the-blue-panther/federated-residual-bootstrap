"""High-power asymptotic study for Local Residual Bootstrap.

Runs with n_mc=500 to provide statistically reliable convergence evidence.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import time

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
    LocalResidualBootstrap,
)
from federated_bootstrap_research.federated.partition import FederatedPartitioner
from federated_bootstrap_research.metrics import (
    compute_coverage,
    compute_bias,
    compute_mse,
)
from federated_bootstrap_research.metrics.wasserstein import compute_wasserstein
from federated_bootstrap_research.metrics.ks_distance import compute_ks_distance


def run_high_power_asymptotic_study(
    n_values: List[int] = [100, 250, 500, 1000, 2500, 5000, 10000],
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 500,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/phase_2_validation/high_power_asymptotic",
) -> Dict[str, Any]:
    """
    Run high-power asymptotic study with large Monte Carlo counts.
    
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
        Number of Monte Carlo simulations per n (default: 500).
    confidence_level : float
        Confidence level for intervals.
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
        Results including asymptotic behavior.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("HIGH-POWER ASYMPTOTIC STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Feature dimension (p):      {p}")
    print(f"  True beta:                  {beta}")
    print(f"  Error sigma:                {sigma}")
    print(f"  Number of sites:            {num_sites}")
    print(f"  Bootstrap iterations:       {n_bootstrap}")
    print(f"  Monte Carlo runs per n:     {n_mc}")
    print(f"  Confidence level:           {confidence_level}")
    print(f"  Distribution:               {distribution}")
    print(f"  Sample sizes:               {n_values}")
    print(f"  Random seed:                {random_state}")
    
    results_by_n = []
    total_sims = sum(n_mc for _ in n_values)
    sim_count = 0
    start_time = time.time()
    
    print("\n" + "-" * 70)
    print("Running high-power asymptotic simulations...")
    print(f"Total simulations: {total_sims}")
    print("-" * 70)
    
    for n_idx, n in enumerate(n_values, 1):
        print(f"\n  Processing n = {n} ({n_idx}/{len(n_values)})")
        n_start = time.time()
        
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
            if (i + 1) % 100 == 0:
                print(f"    Completed {i + 1} / {n_mc} simulations")
            
            sim_count += 1
            
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
                confidence_level=confidence_level,
                random_state=sim_seed,
            )
            central_results = central_bootstrap.fit(X, y)
            central_beta_hats.append(central_results["beta_hat"])
            central_ci_lower.append(central_results["ci_lower"])
            central_ci_upper.append(central_results["ci_upper"])
            central_bootstrap_betas.append(central_results["bootstrap_betas"])
            
            # Partition data
            partitioner = FederatedPartitioner(
                num_sites=num_sites,
                random_state=sim_seed,
            )
            partitions = partitioner.partition(X, y)
            
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
        
        # Compute metrics for centralized
        central_bias = compute_bias(central_beta_hats, beta)
        central_mse = compute_mse(central_beta_hats, beta)
        central_coverage = compute_coverage(central_ci_lower, central_ci_upper, beta)
        central_avg_coverage = np.mean(central_coverage)
        central_variance = np.var(central_beta_hats, axis=0, ddof=1)
        
        # Compute metrics for local
        local_bias = compute_bias(local_beta_hats, beta)
        local_mse = compute_mse(local_beta_hats, beta)
        local_coverage = compute_coverage(local_ci_lower, local_ci_upper, beta)
        local_avg_coverage = np.mean(local_coverage)
        local_variance = np.var(local_beta_hats, axis=0, ddof=1)
        
        # Compute distribution distances
        # Average over Monte Carlo runs for each coefficient
        wasserstein_dist = np.mean([
            compute_wasserstein(central_bootstrap_betas[j], local_bootstrap_betas[j])
            for j in range(n_mc)
        ], axis=0)
        
        ks_dist = np.mean([
            compute_ks_distance(central_bootstrap_betas[j], local_bootstrap_betas[j])
            for j in range(n_mc)
        ], axis=0)
        
        n_time = time.time() - n_start
        
        # Store results
        n_results = {
            "n": n,
            "central_bias": central_bias,
            "central_variance": central_variance,
            "central_mse": central_mse,
            "central_coverage": central_coverage,
            "central_avg_coverage": central_avg_coverage,
            "local_bias": local_bias,
            "local_variance": local_variance,
            "local_mse": local_mse,
            "local_coverage": local_coverage,
            "local_avg_coverage": local_avg_coverage,
            "wasserstein": wasserstein_dist,
            "ks_distance": ks_dist,
            "mean_wasserstein": np.mean(wasserstein_dist),
            "mean_ks": np.mean(ks_dist),
            "time_seconds": n_time,
        }
        results_by_n.append(n_results)
        
        print(f"    Avg coverage (central): {central_avg_coverage:.4f}")
        print(f"    Avg coverage (local):   {local_avg_coverage:.4f}")
        print(f"    Mean Wasserstein:       {n_results['mean_wasserstein']:.6f}")
        print(f"    Mean KS:                {n_results['mean_ks']:.6f}")
        print(f"    Time:                   {n_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s")
    
    # Print summary
    print("\n" + "-" * 70)
    print("Asymptotic Summary")
    print("-" * 70)
    
    print("\n  n | Central Cov | Local Cov | Wasserstein | KS")
    print("  " + "-" * 55)
    for results in results_by_n:
        print(f"  {results['n']:5d} | {results['central_avg_coverage']:.4f}    | {results['local_avg_coverage']:.4f}    | {results['mean_wasserstein']:.6f} | {results['mean_ks']:.6f}")
    
    # Check convergence
    wasserstein_trend = all(
        results_by_n[i+1]['mean_wasserstein'] < results_by_n[i]['mean_wasserstein']
        for i in range(len(results_by_n)-1)
    )
    ks_trend = all(
        results_by_n[i+1]['mean_ks'] < results_by_n[i]['mean_ks']
        for i in range(len(results_by_n)-1)
    )
    
    print(f"\n  Validation:")
    print(f"    Wasserstein decreasing: {'PASSED' if wasserstein_trend else 'FAILED'}")
    print(f"    KS decreasing:          {'PASSED' if ks_trend else 'FAILED'}")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        summary_data = []
        for results in results_by_n:
            row = {
                "n": results["n"],
                "central_avg_coverage": results["central_avg_coverage"],
                "local_avg_coverage": results["local_avg_coverage"],
                "mean_wasserstein": results["mean_wasserstein"],
                "mean_ks": results["mean_ks"],
                "time_seconds": results["time_seconds"],
            }
            for j in range(p):
                row[f"central_bias_{j}"] = results["central_bias"][j]
                row[f"local_bias_{j}"] = results["local_bias"][j]
                row[f"central_variance_{j}"] = results["central_variance"][j]
                row[f"local_variance_{j}"] = results["local_variance"][j]
                row[f"central_mse_{j}"] = results["central_mse"][j]
                row[f"local_mse_{j}"] = results["local_mse"][j]
                row[f"central_coverage_{j}"] = results["central_coverage"][j]
                row[f"local_coverage_{j}"] = results["local_coverage"][j]
                row[f"wasserstein_{j}"] = results["wasserstein"][j]
                row[f"ks_{j}"] = results["ks_distance"][j]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = results_path / f"asymptotic_{distribution}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    
    return {
        "results_by_n": results_by_n,
        "wasserstein_trend": wasserstein_trend,
        "ks_trend": ks_trend,
        "total_time": total_time,
    }


if __name__ == "__main__":
    run_high_power_asymptotic_study(
        n_values=[100, 250, 500, 1000, 2500, 5000, 10000],
        p=5,
        n_bootstrap=500,
        n_mc=100,  # Lower for quick testing, set to 500 for production
        distribution="iid",
        random_state=42,
    )

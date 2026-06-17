"""Asymptotic heterogeneity study for Local Residual Bootstrap.

Investigates whether increasing sample size rescues Local Residual Bootstrap
under cross-site heterogeneity.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation.site_heterogeneity import (
    create_heterogeneous_data,
)
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


def run_asymptotic_heterogeneity_study(
    n_values: List[int] = [100, 250, 500, 1000, 2500, 5000, 10000],
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 50,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    heterogeneity_types: List[str] = ["variance", "feature", "residual_shape", "combined"],
    save_results: bool = True,
    results_dir: str = "results/phase_2_5_validation/asymptotic",
) -> Dict[str, Any]:
    """
    Run asymptotic heterogeneity study.
    
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
    heterogeneity_types : List[str]
        Types of heterogeneity to test.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including asymptotic behavior for each heterogeneity type.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("ASYMPTOTIC HETEROGENEITY STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Feature dimension (p):      {p}")
    print(f"  True beta:                  {beta}")
    print(f"  Error sigma:                {sigma}")
    print(f"  Number of sites:            {num_sites}")
    print(f"  Bootstrap iterations:       {n_bootstrap}")
    print(f"  Monte Carlo runs per n:     {n_mc}")
    print(f"  Confidence level:           {confidence_level}")
    print(f"  Heterogeneity types:        {heterogeneity_types}")
    print(f"  Sample sizes:               {n_values}")
    print(f"  Random seed:                {random_state}")
    
    all_results = {}
    
    for het_type in heterogeneity_types:
        print("\n" + "-" * 70)
        print(f"Heterogeneity Type: {het_type}")
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
                
                # Generate heterogeneous data
                sim_seed = (random_state + hash(het_type) + n_idx * 10000 + i * 1000) if random_state is not None else None
                if sim_seed is not None:
                    sim_seed = abs(sim_seed) % 100000
                
                partitions = create_heterogeneous_data(
                    n=n,
                    p=p,
                    beta=beta,
                    sigma=sigma,
                    random_state=sim_seed,
                    num_sites=num_sites,
                    heterogeneity_type=het_type,
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
                "wasserstein": wasserstein_dist,
                "ks_distance": ks_dist,
                "mean_wasserstein": np.mean(wasserstein_dist),
                "mean_ks": np.mean(ks_dist),
            }
            results_by_n.append(n_results)
            
            print(f"    Avg coverage (central): {central_avg_coverage:.4f}")
            print(f"    Avg coverage (local):   {local_avg_coverage:.4f}")
            print(f"    Mean Wasserstein:       {n_results['mean_wasserstein']:.6f}")
            print(f"    Mean KS:                {n_results['mean_ks']:.6f}")
        
        # Store results for this heterogeneity type
        all_results[het_type] = results_by_n
        
        # Check convergence
        wasserstein_trend = all(
            results_by_n[i+1]['mean_wasserstein'] < results_by_n[i]['mean_wasserstein']
            for i in range(len(results_by_n)-1)
        )
        
        print(f"\n  Convergence for {het_type}:")
        print(f"    Wasserstein decreasing: {'PASSED' if wasserstein_trend else 'FAILED'}")
        
        # Save results
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            summary_data = []
            for results in results_by_n:
                row = {
                    "n": results["n"],
                    "central_avg_coverage": results["central_avg_coverage"],
                    "local_avg_coverage": results["local_avg_coverage"],
                    "mean_wasserstein": results["mean_wasserstein"],
                    "mean_ks": results["mean_ks"],
                }
                for j in range(p):
                    row[f"central_bias_{j}"] = results["central_bias"][j]
                    row[f"local_bias_{j}"] = results["local_bias"][j]
                    row[f"central_mse_{j}"] = results["central_mse"][j]
                    row[f"local_mse_{j}"] = results["local_mse"][j]
                    row[f"central_coverage_{j}"] = results["central_coverage"][j]
                    row[f"local_coverage_{j}"] = results["local_coverage"][j]
                    row[f"wasserstein_{j}"] = results["wasserstein"][j]
                    row[f"ks_{j}"] = results["ks_distance"][j]
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = results_path / f"asymptotic_heterogeneity_{het_type}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"  Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ASYMPTOTIC HETEROGENEITY SUMMARY")
    print("=" * 70)
    print(f"\n{'Type':<15} {'Wasserstein at n=100':<20} {'Wasserstein at n=10000':<22} {'Trend':<10}")
    print("-" * 70)
    for het_type, results in all_results.items():
        w_100 = results[0]['mean_wasserstein']
        w_10000 = results[-1]['mean_wasserstein']
        trend = "Decreasing" if w_10000 < w_100 else "Increasing"
        print(f"{het_type:<15} {w_100:.6f}              {w_10000:.6f}                {trend}")
    
    return all_results


if __name__ == "__main__":
    run_asymptotic_heterogeneity_study(
        n_values=[100, 250, 500, 1000, 2500, 5000, 10000],
        p=5,
        n_bootstrap=500,
        n_mc=15,
        random_state=42,
    )

"""Site count study for Local Residual Bootstrap.

Investigates whether increasing federation degrades the approximation.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
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


def run_site_count_study(
    n_total: int = 10000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_bootstrap: int = 500,
    n_mc: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    site_counts: List[int] = [2, 5, 10, 20, 50, 100],
    save_results: bool = True,
    results_dir: str = "results/phase_2_validation/site_count",
) -> Dict[str, Any]:
    """
    Run site count study.
    
    Parameters
    ----------
    n_total : int
        Total sample size.
    p : int
        Number of features.
    beta : Optional[np.ndarray]
        True coefficients.
    sigma : float
        Error standard deviation.
    n_bootstrap : int
        Number of bootstrap iterations.
    n_mc : int
        Number of Monte Carlo simulations per site count.
    confidence_level : float
        Confidence level for intervals.
    random_state : Optional[int]
        Random seed.
    distribution : str
        Error distribution type.
    site_counts : List[int]
        Number of sites to test.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including comparison across site counts.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("SITE COUNT STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total sample size:      {n_total}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Monte Carlo runs:       {n_mc}")
    print(f"  Confidence level:       {confidence_level}")
    print(f"  Distribution:           {distribution}")
    print(f"  Site counts:            {site_counts}")
    print(f"  Random seed:            {random_state}")
    
    all_results = {}
    
    for M in site_counts:
        print("\n" + "-" * 70)
        print(f"Number of sites: M = {M}")
        print("-" * 70)
        
        # Store results for this M
        local_beta_hats = []
        local_ci_lower = []
        local_ci_upper = []
        bootstrap_betas_all = []
        
        for i in range(n_mc):
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1} / {n_mc} simulations")
            
            # Generate data
            sim_seed = (random_state + M * 10000 + i * 1000) if random_state is not None else None
            X, y = generate_dataset(
                n=n_total,
                p=p,
                beta=beta,
                sigma=sigma,
                random_state=sim_seed,
                distribution=distribution,
            )
            
            # Partition data
            partitioner = FederatedPartitioner(
                num_sites=M,
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
            bootstrap_betas_all.append(local_results["bootstrap_betas"])
        
        # Convert to arrays
        local_beta_hats = np.array(local_beta_hats)
        local_ci_lower = np.array(local_ci_lower)
        local_ci_upper = np.array(local_ci_upper)
        
        # Compute metrics
        local_bias = compute_bias(local_beta_hats, beta)
        local_mse = compute_mse(local_beta_hats, beta)
        local_coverage = compute_coverage(local_ci_lower, local_ci_upper, beta)
        local_avg_coverage = np.mean(local_coverage)
        local_variance = np.var(local_beta_hats, axis=0, ddof=1)
        
        # Store results
        all_results[M] = {
            "bias": local_bias,
            "variance": local_variance,
            "mse": local_mse,
            "coverage": local_coverage,
            "avg_coverage": local_avg_coverage,
            "beta_hats": local_beta_hats,
        }
        
        print(f"\n  Results for M = {M}:")
        print(f"    Average coverage: {local_avg_coverage:.4f}")
        print(f"    Mean bias:        {np.mean(local_bias):.6f}")
        print(f"    Mean MSE:         {np.mean(local_mse):.6f}")
        
        # Save results
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            p_dim = len(beta)
            summary_df = pd.DataFrame({
                "coefficient": [f"beta_{j}" for j in range(p_dim)],
                "bias": local_bias,
                "variance": local_variance,
                "mse": local_mse,
                "coverage": local_coverage,
            })
            summary_path = results_path / f"site_count_M_{M}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SITE COUNT SUMMARY")
    print("=" * 70)
    print(f"\n{'M':<6} {'Coverage':<12} {'Mean Bias':<12} {'Mean MSE':<12}")
    print("-" * 45)
    for M, results in all_results.items():
        print(f"{M:<6} {results['avg_coverage']:.4f}     {np.mean(results['bias']):.6f}   {np.mean(results['mse']):.6f}")
    
    return all_results


if __name__ == "__main__":
    run_site_count_study(
        n_total=10000,
        p=5,
        n_bootstrap=500,
        n_mc=30,  # Lower for quick testing
        random_state=42,
    )

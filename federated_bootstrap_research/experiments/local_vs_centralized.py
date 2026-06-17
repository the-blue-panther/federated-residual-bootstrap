"""Local vs Centralized bootstrap comparison study.

Compares local residual bootstrap against centralized residual bootstrap
under various error distributions.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

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


def run_local_vs_centralized(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 50,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/phase2",
) -> Dict[str, Any]:
    """
    Compare local residual bootstrap against centralized bootstrap.
    
    Parameters
    ----------
    n : int
        Sample size.
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
        Number of Monte Carlo simulations.
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
        Results including comparison metrics.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("LOCAL VS CENTRALIZED BOOTSTRAP COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Number of sites:        {num_sites}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Monte Carlo runs:       {n_mc}")
    print(f"  Confidence level:       {confidence_level}")
    print(f"  Distribution:           {distribution}")
    print(f"  Random seed:            {random_state}")
    
    # Store results
    central_beta_hats = []
    local_beta_hats = []
    central_ci_lower = []
    central_ci_upper = []
    local_ci_lower = []
    local_ci_upper = []
    central_bootstrap_betas = []
    local_bootstrap_betas = []
    
    print("\n" + "-" * 70)
    print("Running Monte Carlo simulations...")
    print("-" * 70)
    
    for i in range(n_mc):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1} / {n_mc} simulations")
        
        # Generate data
        sim_seed = (random_state + i * 1000) if random_state is not None else None
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
        
        # Partition data for federated learning
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
    
    # Compute distribution distances (using last bootstrap distribution)
    # Wasserstein distance
    wasserstein_dist = compute_wasserstein(
        central_bootstrap_betas[-1],
        local_bootstrap_betas[-1],
    )
    
    # KS distance
    ks_dist = compute_ks_distance(
        central_bootstrap_betas[-1],
        local_bootstrap_betas[-1],
    )
    
    # Print results
    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    
    print("\n  Centralized Bootstrap:")
    print(f"    Average coverage: {central_avg_coverage:.4f}")
    print(f"    Mean bias:        {np.mean(central_bias):.6f}")
    print(f"    Mean MSE:         {np.mean(central_mse):.6f}")
    
    print("\n  Local Residual Bootstrap:")
    print(f"    Average coverage: {local_avg_coverage:.4f}")
    print(f"    Mean bias:        {np.mean(local_bias):.6f}")
    print(f"    Mean MSE:         {np.mean(local_mse):.6f}")
    
    print("\n  Distribution Distances:")
    print(f"    Wasserstein distance: {np.mean(wasserstein_dist):.6f}")
    print(f"    KS distance:          {np.mean(ks_dist):.6f}")
    
    # Results dictionary
    results = {
        "n": n,
        "p": p,
        "num_sites": num_sites,
        "n_bootstrap": n_bootstrap,
        "n_mc": n_mc,
        "distribution": distribution,
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
        "wasserstein_distance": wasserstein_dist,
        "ks_distance": ks_dist,
        "central_beta_hats": central_beta_hats,
        "local_beta_hats": local_beta_hats,
        "central_bootstrap_betas": central_bootstrap_betas,
        "local_bootstrap_betas": local_bootstrap_betas,
    }
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        summary_data = {
            "coefficient": [f"beta_{j}" for j in range(p)],
            "central_bias": central_bias,
            "central_mse": central_mse,
            "central_coverage": central_coverage,
            "local_bias": local_bias,
            "local_mse": local_mse,
            "local_coverage": local_coverage,
            "wasserstein": wasserstein_dist,
            "ks_distance": ks_dist,
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = results_path / f"local_vs_centralized_{distribution}_n_{n}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
        
        # Save detailed results
        detailed_df = pd.DataFrame({
            "simulation": np.arange(n_mc),
            **{f"central_beta_{j}": central_beta_hats[:, j] for j in range(p)},
            **{f"local_beta_{j}": local_beta_hats[:, j] for j in range(p)},
        })
        detailed_path = results_path / f"local_vs_centralized_{distribution}_detailed_n_{n}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  Detailed results saved to: {detailed_path}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_local_vs_centralized(
        n=1000,
        p=5,
        n_bootstrap=500,
        n_mc=20,
        distribution="iid",
        random_state=42,
    )

"""Federated coverage study comparing local residual bootstrap against centralized bootstrap."""

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
from federated_bootstrap_research.metrics import compute_coverage


def run_federated_coverage_study(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 500,
    n_mc: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distributions: List[str] = ["iid", "heavy_tailed", "skewed", "heteroscedastic"],
    save_results: bool = True,
    results_dir: str = "results/phase2/coverage",
) -> Dict[str, Any]:
    """
    Run federated coverage study across multiple distributions.
    
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
        Number of Monte Carlo simulations per distribution.
    confidence_level : float
        Confidence level for intervals.
    random_state : Optional[int]
        Random seed.
    distributions : List[str]
        Error distributions to test.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including coverage comparison.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("FEDERATED COVERAGE STUDY")
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
    print(f"  Distributions:          {distributions}")
    print(f"  Random seed:            {random_state}")
    
    all_results = {}
    
    for dist_idx, distribution in enumerate(distributions, 1):
        print(f"\n" + "-" * 70)
        print(f"Distribution {dist_idx}/{len(distributions)}: {distribution}")
        print("-" * 70)
        
        # Store results for this distribution
        central_ci_lower = []
        central_ci_upper = []
        local_ci_lower = []
        local_ci_upper = []
        
        for i in range(n_mc):
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1} / {n_mc} simulations")
            
            # Generate data
            sim_seed = (random_state + dist_idx * 10000 + i * 1000) if random_state is not None else None
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
            central_ci_lower.append(central_results["ci_lower"])
            central_ci_upper.append(central_results["ci_upper"])
            
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
            local_ci_lower.append(local_results["ci_lower"])
            local_ci_upper.append(local_results["ci_upper"])
        
        # Convert to arrays
        central_ci_lower = np.array(central_ci_lower)
        central_ci_upper = np.array(central_ci_upper)
        local_ci_lower = np.array(local_ci_lower)
        local_ci_upper = np.array(local_ci_upper)
        
        # Compute coverage
        central_coverage = compute_coverage(central_ci_lower, central_ci_upper, beta)
        local_coverage = compute_coverage(local_ci_lower, local_ci_upper, beta)
        
        central_avg = np.mean(central_coverage)
        local_avg = np.mean(local_coverage)
        coverage_diff = local_avg - central_avg
        
        print(f"\n  Results for {distribution}:")
        print(f"    Centralized coverage: {central_avg:.4f}")
        print(f"    Local coverage:       {local_avg:.4f}")
        print(f"    Difference:           {coverage_diff:+.4f}")
        
        # Store results
        all_results[distribution] = {
            "central_coverage": central_coverage,
            "local_coverage": local_coverage,
            "central_avg": central_avg,
            "local_avg": local_avg,
            "coverage_diff": coverage_diff,
        }
        
        # Save results
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            summary_df = pd.DataFrame({
                "coefficient": [f"beta_{j}" for j in range(p)],
                "central_coverage": central_coverage,
                "local_coverage": local_coverage,
            })
            summary_path = results_path / f"coverage_{distribution}.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved to: {summary_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    print(f"\n{'Distribution':<15} {'Central':<10} {'Local':<10} {'Diff':<10}")
    print("-" * 45)
    for dist, res in all_results.items():
        print(f"{dist:<15} {res['central_avg']:.4f}     {res['local_avg']:.4f}     {res['coverage_diff']:+.4f}")
    
    return all_results


if __name__ == "__main__":
    run_federated_coverage_study(
        n=1000,
        p=5,
        n_bootstrap=500,
        n_mc=50,
        random_state=42,
    )

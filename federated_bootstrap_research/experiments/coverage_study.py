"""Coverage study for centralized residual bootstrap.

Repeated Monte Carlo simulation to validate coverage probabilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)
from federated_bootstrap_research.metrics import compute_coverage


def run_coverage_study(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_bootstrap: int = 500,
    n_mc: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/coverage",
) -> Dict[str, Any]:
    """
    Run coverage study.
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    beta : Optional[np.ndarray]
        True coefficients. Default: [2.5, -1.8, 3.2, 0.5, -0.7]
    sigma : float
        Error standard deviation.
    n_bootstrap : int
        Number of bootstrap iterations.
    n_mc : int
        Number of Monte Carlo simulations.
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
        Results including coverage rates and detailed data.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("COVERAGE STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Monte Carlo runs:       {n_mc}")
    print(f"  Confidence level:       {confidence_level}")
    print(f"  Random seed:            {random_state}")
    
    # Store results
    beta_hats = []
    bootstrap_betas_list = []
    ci_lower_all = []
    ci_upper_all = []
    
    # Main loop
    print("\n" + "-" * 70)
    print("Running Monte Carlo simulations...")
    print("-" * 70)
    
    for i in range(n_mc):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1} / {n_mc} simulations")
        
        # Generate data with different random state for each simulation
        sim_seed = (random_state + i * 1000) if random_state is not None else None
        X, y = generate_dataset(
            n=n,
            p=p,
            beta=beta,
            sigma=sigma,
            random_state=sim_seed,
            distribution=distribution,
        )
        
        # Fit bootstrap
        bootstrap = CentralizedResidualBootstrap(
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=sim_seed,
        )
        results = bootstrap.fit(X, y)
        
        beta_hats.append(results["beta_hat"])
        bootstrap_betas_list.append(results["bootstrap_betas"])
        ci_lower_all.append(results["ci_lower"])
        ci_upper_all.append(results["ci_upper"])
    
    # Convert to arrays
    beta_hats = np.array(beta_hats)
    ci_lower_all = np.array(ci_lower_all)
    ci_upper_all = np.array(ci_upper_all)
    
    # Compute coverage
    coverage = compute_coverage(ci_lower_all, ci_upper_all, beta)
    avg_coverage = np.mean(coverage)
    
    # Compute bias, variance, MSE
    bias = np.mean(beta_hats, axis=0) - beta
    variance = np.var(beta_hats, axis=0, ddof=1)
    mse = np.mean((beta_hats - beta) ** 2, axis=0)
    
    # Results
    results = {
        "n": n,
        "p": p,
        "n_bootstrap": n_bootstrap,
        "n_mc": n_mc,
        "confidence_level": confidence_level,
        "coverage": coverage,
        "avg_coverage": avg_coverage,
        "bias": bias,
        "variance": variance,
        "mse": mse,
        "beta_hats": beta_hats,
        "ci_lower_all": ci_lower_all,
        "ci_upper_all": ci_upper_all,
    }
    
    # Print results
    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"\n  Average coverage: {avg_coverage:.4f}")
    print(f"  Expected coverage: {confidence_level:.4f}")
    print(f"  Difference: {avg_coverage - confidence_level:+.4f}")
    
    print("\n  Per-coefficient coverage:")
    for j in range(p):
        print(f"    beta_{j}: {coverage[j]:.4f}")
    
    print("\n  Per-coefficient bias:")
    for j in range(p):
        print(f"    beta_{j}: {bias[j]:.6f}")
    
    print("\n  Per-coefficient MSE:")
    for j in range(p):
        print(f"    beta_{j}: {mse[j]:.6f}")
    
    # Validate
    is_valid = (0.93 <= avg_coverage <= 0.97)
    print(f"\n  Validation: {'PASSED' if is_valid else 'FAILED'}")
    if is_valid:
        print(f"    Coverage {avg_coverage:.4f} within acceptable range [0.93, 0.97]")
    else:
        print(f"    Coverage {avg_coverage:.4f} outside acceptable range [0.93, 0.97]")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_df = pd.DataFrame({
            "coefficient": [f"beta_{j}" for j in range(p)],
            "true_value": beta,
            "mean_estimate": np.mean(beta_hats, axis=0),
            "bias": bias,
            "variance": variance,
            "mse": mse,
            "coverage": coverage,
        })
        summary_path = results_path / f"coverage_{distribution}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
        
        # Save detailed results
        detailed_df = pd.DataFrame({
            "simulation": np.arange(n_mc),
            **{f"beta_{j}_hat": beta_hats[:, j] for j in range(p)},
            **{f"ci_lower_{j}": ci_lower_all[:, j] for j in range(p)},
            **{f"ci_upper_{j}": ci_upper_all[:, j] for j in range(p)},
        })
        detailed_path = results_path / f"coverage_{distribution}_detailed.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  Detailed results saved to: {detailed_path}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    # Run coverage study with default parameters
    run_coverage_study(
        n=1000,
        p=5,
        n_bootstrap=500,
        n_mc=1000,
        random_state=42,
    )

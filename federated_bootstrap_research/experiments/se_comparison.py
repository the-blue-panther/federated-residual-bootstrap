"""Standard error comparison study.

Compares bootstrap standard errors against theoretical standard errors.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)
from federated_bootstrap_research.metrics import compute_theoretical_se


def run_se_comparison(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_bootstrap: int = 500,
    n_mc: int = 100,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/se_comparison",
) -> Dict[str, Any]:
    """
    Run standard error comparison study.
    
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
    n_bootstrap : int
        Number of bootstrap iterations.
    n_mc : int
        Number of Monte Carlo simulations.
    random_state : Optional[int]
        Random seed.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including SE comparison.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("STANDARD ERROR COMPARISON STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Monte Carlo runs:       {n_mc}")
    print(f"  Random seed:            {random_state}")
    
    # Store results
    bootstrap_se_list = []
    theoretical_se_list = []
    
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
        
        # Fit bootstrap
        bootstrap = CentralizedResidualBootstrap(
            n_bootstrap=n_bootstrap,
            confidence_level=0.95,
            random_state=sim_seed,
        )
        results = bootstrap.fit(X, y)
        
        # Compute theoretical SE
        theoretical_se = compute_theoretical_se(X, sigma)
        
        bootstrap_se_list.append(results["bootstrap_se"])
        theoretical_se_list.append(theoretical_se)
    
    # Convert to arrays
    bootstrap_se = np.array(bootstrap_se_list)
    theoretical_se = np.array(theoretical_se_list)
    
    # Compute average SEs
    mean_bootstrap_se = np.mean(bootstrap_se, axis=0)
    mean_theoretical_se = np.mean(theoretical_se, axis=0)
    
    # Compute relative errors
    relative_errors = np.abs(mean_bootstrap_se - mean_theoretical_se) / mean_theoretical_se
    
    # Results
    results = {
        "n": n,
        "p": p,
        "n_bootstrap": n_bootstrap,
        "n_mc": n_mc,
        "mean_bootstrap_se": mean_bootstrap_se,
        "mean_theoretical_se": mean_theoretical_se,
        "relative_errors": relative_errors,
        "bootstrap_se_all": bootstrap_se,
        "theoretical_se_all": theoretical_se,
    }
    
    # Print results
    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    
    print("\n  Standard error comparison:")
    for j in range(p):
        print(f"    beta_{j}:")
        print(f"      Bootstrap SE:  {mean_bootstrap_se[j]:.6f}")
        print(f"      Theoretical SE: {mean_theoretical_se[j]:.6f}")
        print(f"      Relative error: {relative_errors[j]*100:.2f}%")
    
    max_rel_error = np.max(relative_errors)
    avg_rel_error = np.mean(relative_errors)
    
    print(f"\n  Average relative error: {avg_rel_error*100:.2f}%")
    print(f"  Maximum relative error: {max_rel_error*100:.2f}%")
    
    # Validate
    is_valid = np.all(relative_errors < 0.10)
    print(f"\n  Validation: {'PASSED' if is_valid else 'FAILED'}")
    if is_valid:
        print(f"    All relative errors < 10%")
    else:
        print(f"    Some relative errors > 10%")
        for j in range(p):
            if relative_errors[j] >= 0.10:
                print(f"      beta_{j}: {relative_errors[j]*100:.2f}%")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        summary_df = pd.DataFrame({
            "coefficient": [f"beta_{j}" for j in range(p)],
            "bootstrap_se": mean_bootstrap_se,
            "theoretical_se": mean_theoretical_se,
            "relative_error": relative_errors,
            "relative_error_pct": relative_errors * 100,
        })
        summary_path = results_path / f"se_comparison_{distribution}_n_{n}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
        
        # Save detailed results
        detailed_df = pd.DataFrame({
            "simulation": np.arange(n_mc),
            **{f"bootstrap_se_{j}": bootstrap_se[:, j] for j in range(p)},
            **{f"theoretical_se_{j}": theoretical_se[:, j] for j in range(p)},
        })
        detailed_path = results_path / f"se_comparison_{distribution}_detailed_n_{n}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"  Detailed results saved to: {detailed_path}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_se_comparison(
        n=1000,
        p=5,
        n_bootstrap=500,
        n_mc=100,
        random_state=42,
    )

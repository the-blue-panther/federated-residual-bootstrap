"""Asymptotic study for centralized residual bootstrap.

Investigates bias, variance, MSE, and coverage as n increases.
Supports multiple error distributions: iid, heavy_tailed, skewed, heteroscedastic.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)
from federated_bootstrap_research.metrics import compute_coverage


def run_asymptotic_study(
    n_values: List[int] = [100, 500, 1000, 5000, 10000],
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_bootstrap: int = 500,
    n_mc: int = 50,
    confidence_level: float = 0.95,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_results: bool = True,
    results_dir: str = "results/asymptotic",
) -> Dict[str, Any]:
    """
    Run asymptotic study across different sample sizes.
    
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
        Results including asymptotic behavior.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("ASYMPTOTIC STUDY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Feature dimension (p):      {p}")
    print(f"  True beta:                  {beta}")
    print(f"  Error sigma:                {sigma}")
    print(f"  Bootstrap iterations:       {n_bootstrap}")
    print(f"  Monte Carlo runs per n:     {n_mc}")
    print(f"  Confidence level:           {confidence_level}")
    print(f"  Sample sizes:               {n_values}")
    print(f"  Random seed:                {random_state}")
    
    results_by_n = []
    
    print("\n" + "-" * 70)
    print("Running asymptotic simulations...")
    print("-" * 70)
    
    for n_idx, n in enumerate(n_values, 1):
        print(f"\n  Processing n = {n} ({n_idx}/{len(n_values)})")
        
        # Store results for this n
        beta_hats = []
        ci_lower_all = []
        ci_upper_all = []
        
        for i in range(n_mc):
            # Generate data
            sim_seed = (random_state + n_idx * 10000 + i * 1000) if random_state is not None else None
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
            ci_lower_all.append(results["ci_lower"])
            ci_upper_all.append(results["ci_upper"])
        
        # Convert to arrays
        beta_hats = np.array(beta_hats)
        ci_lower_all = np.array(ci_lower_all)
        ci_upper_all = np.array(ci_upper_all)
        
        # Compute metrics
        bias = np.mean(beta_hats, axis=0) - beta
        variance = np.var(beta_hats, axis=0, ddof=1)
        mse = np.mean((beta_hats - beta) ** 2, axis=0)
        coverage = compute_coverage(ci_lower_all, ci_upper_all, beta)
        avg_coverage = np.mean(coverage)
        
        # Store results
        n_results = {
            "n": n,
            "bias": bias,
            "variance": variance,
            "mse": mse,
            "coverage": coverage,
            "avg_coverage": avg_coverage,
            "mean_bias": np.mean(bias),
            "mean_absolute_bias": np.mean(np.abs(bias)),
            "mean_variance": np.mean(variance),
            "mean_mse": np.mean(mse),
        }
        results_by_n.append(n_results)
        
        print(f"    Avg coverage: {avg_coverage:.4f}")
        print(f"    Mean bias:    {n_results['mean_bias']:.6f}")
        print(f"    Mean MSE:     {n_results['mean_mse']:.6f}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("Asymptotic Summary")
    print("-" * 70)
    
    print("\n  n | Avg Coverage | Mean Bias | Mean MSE")
    print("  " + "-" * 40)
    for results in results_by_n:
        print(f"  {results['n']:5d} | {results['avg_coverage']:.4f}    | {results['mean_bias']:.6f} | {results['mean_mse']:.6f}")
    
    # Validate using absolute bias
    # Check if absolute bias decreases and MSE decreases
    abs_bias_trend = all(
        results_by_n[i+1]['mean_absolute_bias'] < results_by_n[i]['mean_absolute_bias'] 
        for i in range(len(results_by_n)-1)
    )
    mse_trend = all(
        results_by_n[i+1]['mean_mse'] < results_by_n[i]['mean_mse'] 
        for i in range(len(results_by_n)-1)
    )
    
    print(f"\n  Validation:")
    print(f"    Absolute bias decreasing: {'PASSED' if abs_bias_trend else 'FAILED'}")
    print(f"    MSE decreasing:           {'PASSED' if mse_trend else 'FAILED'}")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        summary_data = []
        for results in results_by_n:
            row = {
                "n": results["n"],
                "avg_coverage": results["avg_coverage"],
                "mean_bias": results["mean_bias"],
                "mean_variance": results["mean_variance"],
                "mean_mse": results["mean_mse"],
            }
            for j in range(p):
                row[f"bias_beta_{j}"] = results["bias"][j]
                row[f"variance_beta_{j}"] = results["variance"][j]
                row[f"mse_beta_{j}"] = results["mse"][j]
                row[f"coverage_beta_{j}"] = results["coverage"][j]
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = results_path / "asymptotic_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved to: {summary_path}")
    
    print("\n" + "=" * 70)
    
    return {
        "results_by_n": results_by_n,
        "abs_bias_trend": abs_bias_trend,
        "mse_trend": mse_trend,
    }


if __name__ == "__main__":
    run_asymptotic_study(
        n_values=[100, 500, 1000, 5000, 10000],
        p=5,
        n_bootstrap=500,
        n_mc=50,
        random_state=42,
    )

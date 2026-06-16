"""Runtime benchmark for centralized residual bootstrap.

Measures runtime versus n, p, and B.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)


def run_runtime_benchmark(
    n_values: List[int] = [100, 500, 1000, 5000],
    p_values: List[int] = [2, 5, 10, 20],
    B_values: List[int] = [100, 500, 1000],
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_repeats: int = 3,
    random_state: Optional[int] = 42,
    save_results: bool = True,
    results_dir: str = "results/runtime",
) -> Dict[str, Any]:
    """
    Run runtime benchmark.
    
    Parameters
    ----------
    n_values : List[int]
        Sample sizes to test.
    p_values : List[int]
        Feature dimensions to test.
    B_values : List[int]
        Bootstrap iterations to test.
    beta : Optional[np.ndarray]
        True coefficients.
    sigma : float
        Error standard deviation.
    n_repeats : int
        Number of repeats for each configuration.
    random_state : Optional[int]
        Random seed.
    save_results : bool
        Whether to save results to CSV.
    results_dir : str
        Directory to save results.
    
    Returns
    -------
    Dict[str, Any]
        Results including runtime measurements.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("RUNTIME BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Error sigma:            {sigma}")
    print(f"  Repeats per config:     {n_repeats}")
    print(f"  Random seed:            {random_state}")
    print(f"  n values:               {n_values}")
    print(f"  p values:               {p_values}")
    print(f"  B values:               {B_values}")
    
    results = []
    
    print("\n" + "-" * 70)
    print("Running benchmarks...")
    print("-" * 70)
    
    total_configs = len(n_values) * len(p_values) * len(B_values)
    config_count = 0
    
    for n in n_values:
        for p in p_values:
            # Adjust beta for different p
            if p != len(beta):
                beta_p = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
                if p <= 5:
                    beta_p = beta_p[:p]
                else:
                    beta_p = np.concatenate([beta_p, np.zeros(p - 5)])
            else:
                beta_p = beta
            
            for B in B_values:
                config_count += 1
                print(f"  [{config_count}/{total_configs}] n={n}, p={p}, B={B}")
                
                # Generate data
                X, y = generate_dataset(
                    n=n,
                    p=p,
                    beta=beta_p,
                    sigma=sigma,
                    random_state=random_state,
                )
                
                # Time the bootstrap
                runtimes = []
                for repeat in range(n_repeats):
                    bootstrap = CentralizedResidualBootstrap(
                        n_bootstrap=B,
                        confidence_level=0.95,
                        random_state=random_state + repeat * 1000 if random_state else None,
                    )
                    
                    start_time = time.time()
                    _ = bootstrap.fit(X, y)
                    end_time = time.time()
                    
                    runtimes.append(end_time - start_time)
                
                # Store results
                avg_runtime = np.mean(runtimes)
                std_runtime = np.std(runtimes)
                
                results.append({
                    "n": n,
                    "p": p,
                    "B": B,
                    "avg_runtime_seconds": avg_runtime,
                    "std_runtime_seconds": std_runtime,
                    "min_runtime": np.min(runtimes),
                    "max_runtime": np.max(runtimes),
                    "runtime_per_bootstrap": avg_runtime / B,
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "-" * 70)
    print("Benchmark Summary")
    print("-" * 70)
    
    # Group by n
    print("\n  Runtime by sample size (n):")
    for n in n_values:
        subset = results_df[results_df["n"] == n]
        avg_time = subset["avg_runtime_seconds"].mean()
        print(f"    n={n}: {avg_time:.4f} seconds")
    
    # Group by p
    print("\n  Runtime by feature dimension (p):")
    for p in p_values:
        subset = results_df[results_df["p"] == p]
        avg_time = subset["avg_runtime_seconds"].mean()
        print(f"    p={p}: {avg_time:.4f} seconds")
    
    # Group by B
    print("\n  Runtime by bootstrap iterations (B):")
    for B in B_values:
        subset = results_df[results_df["B"] == B]
        avg_time = subset["avg_runtime_seconds"].mean()
        print(f"    B={B}: {avg_time:.4f} seconds")
    
    # Save results
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Full results
        full_path = results_path / "runtime_benchmark_full.csv"
        results_df.to_csv(full_path, index=False)
        print(f"\n  Full results saved to: {full_path}")
        
        # Summary by n
        summary_n = results_df.groupby("n").agg({
            "avg_runtime_seconds": ["mean", "std", "min", "max"]
        }).round(4)
        summary_n_path = results_path / "runtime_summary_n.csv"
        summary_n.to_csv(summary_n_path)
        print(f"  Summary by n saved to: {summary_n_path}")
        
        # Summary by p
        summary_p = results_df.groupby("p").agg({
            "avg_runtime_seconds": ["mean", "std", "min", "max"]
        }).round(4)
        summary_p_path = results_path / "runtime_summary_p.csv"
        summary_p.to_csv(summary_p_path)
        print(f"  Summary by p saved to: {summary_p_path}")
        
        # Summary by B
        summary_B = results_df.groupby("B").agg({
            "avg_runtime_seconds": ["mean", "std", "min", "max"]
        }).round(4)
        summary_B_path = results_path / "runtime_summary_B.csv"
        summary_B.to_csv(summary_B_path)
        print(f"  Summary by B saved to: {summary_B_path}")
    
    print("\n" + "=" * 70)
    
    return {
        "results_df": results_df,
        "results": results,
    }


if __name__ == "__main__":
    run_runtime_benchmark(
        n_values=[100, 500, 1000, 5000],
        p_values=[2, 5, 10],
        B_values=[100, 500, 1000],
        n_repeats=3,
        random_state=42,
    )

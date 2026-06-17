"""Convergence plots for Local Residual Bootstrap validation.

Generates publication-quality plots for all validation experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any


def create_convergence_plots(
    results: pd.DataFrame,
    title: str = "Convergence Analysis",
    save_path: Optional[Path] = None,
) -> None:
    """
    Create convergence plots from results DataFrame.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results with columns: n, central_avg_coverage, local_avg_coverage,
        mean_wasserstein, mean_ks
    title : str
        Plot title
    save_path : Optional[Path]
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    n = results["n"].values
    
    # 1. Coverage vs n
    ax = axes[0, 0]
    ax.plot(n, results["central_avg_coverage"], 'b-o', label='Centralized', linewidth=2, markersize=8)
    ax.plot(n, results["local_avg_coverage"], 'g-s', label='Local Residual', linewidth=2, markersize=8)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage vs Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Wasserstein Distance vs n
    ax = axes[0, 1]
    ax.plot(n, results["mean_wasserstein"], 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Wasserstein Distance vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 3. KS Distance vs n
    ax = axes[1, 0]
    ax.plot(n, results["mean_ks"], 'purple-o', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('KS Distance')
    ax.set_title('Kolmogorov-Smirnov Distance vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 4. Coverage Difference
    ax = axes[1, 1]
    diff = results["local_avg_coverage"] - results["central_avg_coverage"]
    ax.plot(n, diff, 'b-o', linewidth=2, markersize=8)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Coverage Difference (Local - Central)')
    ax.set_title('Coverage Difference vs Sample Size')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    plt.close()


def create_imbalance_plots(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Create site imbalance plots.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Results from site imbalance study
    save_path : Optional[Path]
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = list(results.keys())
    coverages = [results[s]["avg_coverage"] for s in scenarios]
    mses = [np.mean(results[s]["mse"]) for s in scenarios]
    
    # 1. Coverage by scenario
    ax = axes[0]
    bars = ax.bar(scenarios, coverages, color=['blue', 'green', 'red'])
    ax.axhline(0.95, color='black', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage by Imbalance Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. MSE by scenario
    ax = axes[1]
    ax.bar(scenarios, mses, color=['blue', 'green', 'red'])
    ax.set_ylabel('Mean MSE')
    ax.set_title('MSE by Imbalance Level')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Site Imbalance Study Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    plt.close()


def create_site_count_plots(
    results: Dict[int, Dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Create site count plots.
    
    Parameters
    ----------
    results : Dict[int, Dict]
        Results from site count study
    save_path : Optional[Path]
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    M_values = sorted(results.keys())
    coverages = [results[M]["avg_coverage"] for M in M_values]
    mses = [np.mean(results[M]["mse"]) for M in M_values]
    
    # 1. Coverage vs M
    ax = axes[0]
    ax.plot(M_values, coverages, 'b-o', linewidth=2, markersize=8)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.set_xlabel('Number of Sites (M)')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage vs Number of Sites')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. MSE vs M
    ax = axes[1]
    ax.plot(M_values, mses, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Sites (M)')
    ax.set_ylabel('Mean MSE')
    ax.set_title('MSE vs Number of Sites')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.suptitle('Site Count Study Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    plt.close()


def create_distribution_robustness_plots(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Create distribution robustness plots.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Results from distribution robustness study
    save_path : Optional[Path]
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    distributions = list(results.keys())
    central_cov = [results[d]["central_avg_coverage"] for d in distributions]
    local_cov = [results[d]["local_avg_coverage"] for d in distributions]
    wasserstein = [results[d]["mean_wasserstein"] for d in distributions]
    
    x = np.arange(len(distributions))
    width = 0.35
    
    # 1. Coverage comparison
    ax = axes[0]
    ax.bar(x - width/2, central_cov, width, label='Centralized', color='blue')
    ax.bar(x + width/2, local_cov, width, label='Local Residual', color='green')
    ax.axhline(0.95, color='red', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.set_xlabel('Distribution')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage by Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(distributions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Wasserstein distance
    ax = axes[1]
    ax.bar(distributions, wasserstein, color='purple')
    ax.set_xlabel('Distribution')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Wasserstein Distance by Distribution')
    ax.set_xticklabels(distributions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Robustness Study Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")
    
    plt.close()

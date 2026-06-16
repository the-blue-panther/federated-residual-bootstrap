"""Distribution diagnostics for centralized residual bootstrap.

Generates histograms, QQ plots, and normal overlays for bootstrap distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
from typing import Optional

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)


def run_distribution_diagnostics(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = 42,
    save_plots: bool = True,
    results_dir: str = "results/diagnostics",
) -> None:
    """
    Generate distribution diagnostics plots.
    
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
    random_state : Optional[int]
        Random seed.
    save_plots : bool
        Whether to save plots to files.
    results_dir : str
        Directory to save plots.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Random seed:            {random_state}")
    
    # Generate data
    print("\n" + "-" * 70)
    print("Generating data and fitting bootstrap...")
    print("-" * 70)
    
    X, y = generate_dataset(
        n=n,
        p=p,
        beta=beta,
        sigma=sigma,
        random_state=random_state,
    )
    
    bootstrap = CentralizedResidualBootstrap(
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state,
    )
    results = bootstrap.fit(X, y)
    
    bootstrap_betas = results["bootstrap_betas"]
    beta_hat = results["beta_hat"]
    
    print(f"  Bootstrap betas shape: {bootstrap_betas.shape}")
    
    # Create plots
    print("\n" + "-" * 70)
    print("Generating diagnostic plots...")
    print("-" * 70)
    
    if save_plots:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
    
    # Plot for each coefficient
    for j in range(min(p, 5)):  # Limit to first 5 coefficients for clarity
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Get bootstrap distribution for this coefficient
        beta_star_j = bootstrap_betas[:, j]
        
        # 1. Histogram with normal overlay
        axes[0].hist(beta_star_j, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Overlay normal distribution
        mu, std = np.mean(beta_star_j), np.std(beta_star_j)
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        normal_pdf = stats.norm.pdf(x, mu, std)
        axes[0].plot(x, normal_pdf, 'r-', linewidth=2, label='Normal fit')
        axes[0].axvline(beta_hat[j], color='green', linestyle='--', linewidth=2, label=f'beta_hat = {beta_hat[j]:.3f}')
        axes[0].axvline(beta[j], color='orange', linestyle='--', linewidth=2, label=f'true beta = {beta[j]:.3f}')
        axes[0].set_xlabel(f'beta_{j}*')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'beta_{j} Distribution')
        axes[0].legend()
        
        # 2. QQ plot
        stats.probplot(beta_star_j, dist="norm", plot=axes[1])
        axes[1].set_title(f'beta_{j} QQ Plot')
        
        # 3. Density plot
        axes[2].hist(beta_star_j, bins=30, density=True, alpha=0.5, color='blue', edgecolor='black')
        axes[2].plot(x, normal_pdf, 'r-', linewidth=2, label='Normal')
        axes[2].set_xlabel(f'beta_{j}*')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'beta_{j} Density Overlay')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = results_path / f"beta_{j}_diagnostics.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
        
        plt.close()
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics")
    print("-" * 70)
    
    for j in range(p):
        beta_star_j = bootstrap_betas[:, j]
        mean_star = np.mean(beta_star_j)
        std_star = np.std(beta_star_j)
        skewness = stats.skew(beta_star_j)
        kurtosis = stats.kurtosis(beta_star_j)
        
        print(f"\n  beta_{j}:")
        print(f"    Mean:      {mean_star:.6f}")
        print(f"    Std:       {std_star:.6f}")
        print(f"    Skewness:  {skewness:.4f}")
        print(f"    Kurtosis:  {kurtosis:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_distribution_diagnostics(
        n=1000,
        p=5,
        n_bootstrap=1000,
        random_state=42,
    )

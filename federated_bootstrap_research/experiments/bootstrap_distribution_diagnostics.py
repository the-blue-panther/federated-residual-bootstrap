"""Bootstrap distribution diagnostics for Local Residual Bootstrap.

Generates histograms, QQ plots, and ECDF comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Optional, List
from pathlib import Path

from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
    LocalResidualBootstrap,
)
from federated_bootstrap_research.federated.partition import FederatedPartitioner


def run_bootstrap_distribution_diagnostics(
    n: int = 1000,
    p: int = 5,
    beta: Optional[np.ndarray] = None,
    sigma: float = 1.5,
    num_sites: int = 3,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = 42,
    distribution: str = "iid",
    save_plots: bool = True,
    results_dir: str = "results/phase_2_validation/diagnostics",
) -> None:
    """
    Generate bootstrap distribution diagnostics plots.
    
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
    random_state : Optional[int]
        Random seed.
    distribution : str
        Error distribution type.
    save_plots : bool
        Whether to save plots to files.
    results_dir : str
        Directory to save plots.
    """
    if beta is None:
        beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    
    print("=" * 70)
    print("BOOTSTRAP DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Number of sites:        {num_sites}")
    print(f"  Bootstrap iterations:   {n_bootstrap}")
    print(f"  Distribution:           {distribution}")
    print(f"  Random seed:            {random_state}")
    
    # Generate data
    print("\n" + "-" * 70)
    print("Generating data and fitting bootstrap models...")
    print("-" * 70)
    
    X, y = generate_dataset(
        n=n,
        p=p,
        beta=beta,
        sigma=sigma,
        random_state=random_state,
        distribution=distribution,
    )
    
    # Centralized bootstrap
    central_bootstrap = CentralizedResidualBootstrap(
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state,
    )
    central_results = central_bootstrap.fit(X, y)
    central_betas = central_results["bootstrap_betas"]
    central_beta_hat = central_results["beta_hat"]
    
    print(f"  Centralized bootstrap: {central_betas.shape}")
    
    # Partition data
    partitioner = FederatedPartitioner(
        num_sites=num_sites,
        random_state=random_state,
    )
    partitions = partitioner.partition(X, y)
    
    # Local residual bootstrap
    local_bootstrap = LocalResidualBootstrap(
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state,
    )
    local_results = local_bootstrap.fit(partitions)
    local_betas = local_results["bootstrap_betas"]
    local_beta_hat = local_results["beta_hat"]
    
    print(f"  Local residual bootstrap: {local_betas.shape}")
    
    if save_plots:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
    
    # Plot for each coefficient (limit to first 5 for clarity)
    print("\n" + "-" * 70)
    print("Generating diagnostic plots...")
    print("-" * 70)
    
    for j in range(min(p, 5)):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get bootstrap distributions
        central_beta_j = central_betas[:, j]
        local_beta_j = local_betas[:, j]
        
        # 1. Histogram - Centralized
        axes[0, 0].hist(central_beta_j, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        mu, std = np.mean(central_beta_j), np.std(central_beta_j)
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
        axes[0, 0].axvline(central_beta_hat[j], color='green', linestyle='--', linewidth=2, label=f'beta_hat={central_beta_hat[j]:.3f}')
        axes[0, 0].axvline(beta[j], color='orange', linestyle='--', linewidth=2, label=f'true={beta[j]:.3f}')
        axes[0, 0].set_xlabel(f'beta_{j}*')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title(f'Centralized - beta_{j}')
        axes[0, 0].legend()
        
        # 2. Histogram - Local
        axes[0, 1].hist(local_beta_j, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        mu, std = np.mean(local_beta_j), np.std(local_beta_j)
        x = np.linspace(mu - 4*std, mu + 4*std, 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
        axes[0, 1].axvline(local_beta_hat[j], color='blue', linestyle='--', linewidth=2, label=f'beta_hat={local_beta_hat[j]:.3f}')
        axes[0, 1].axvline(beta[j], color='orange', linestyle='--', linewidth=2, label=f'true={beta[j]:.3f}')
        axes[0, 1].set_xlabel(f'beta_{j}*')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title(f'Local - beta_{j}')
        axes[0, 1].legend()
        
        # 3. QQ plot - Centralized
        stats.probplot(central_beta_j, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title(f'Centralized beta_{j} QQ Plot')
        
        # 4. QQ plot - Local
        stats.probplot(local_beta_j, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Local beta_{j} QQ Plot')
        
        # 5. ECDF comparison
        ax = axes[1, 1]
        ax.hist(central_beta_j, bins=30, density=True, alpha=0.3, color='blue', label='Centralized')
        ax.hist(local_beta_j, bins=30, density=True, alpha=0.3, color='green', label='Local')
        ax.set_xlabel(f'beta_{j}*')
        ax.set_ylabel('Density')
        ax.set_title(f'ECDF Comparison - beta_{j}')
        ax.legend()
        
        # 6. Difference plot
        ax = axes[1, 2]
        # Compute ECDFs
        x_sorted = np.sort(np.concatenate([central_beta_j, local_beta_j]))
        central_ecdf = np.searchsorted(np.sort(central_beta_j), x_sorted) / len(central_beta_j)
        local_ecdf = np.searchsorted(np.sort(local_beta_j), x_sorted) / len(local_beta_j)
        ax.plot(x_sorted, central_ecdf - local_ecdf, 'b-', linewidth=2)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel(f'beta_{j}*')
        ax.set_ylabel('ECDF Difference')
        ax.set_title(f'ECDF Difference - beta_{j}')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = results_path / f"diagnostics_beta_{j}_{distribution}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {plot_path}")
        
        plt.close()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_bootstrap_distribution_diagnostics(
        n=1000,
        p=5,
        n_bootstrap=1000,
        random_state=42,
    )

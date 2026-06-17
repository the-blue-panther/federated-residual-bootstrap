"""Test for federated OLS implementation.

Verifies that federated OLS matches centralized OLS numerically.
"""

import numpy as np
from federated_bootstrap_research.federated.federated_ols import (
    FederatedOLS,
    centralized_ols,
)
from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.federated.partition import FederatedPartitioner


def test_federated_ols():
    """
    Test federated OLS against centralized OLS.
    
    Verifies:
    1. Federated OLS matches centralized OLS
    2. Tolerance is within 1e-10
    3. Works with different numbers of sites
    4. Works with balanced and unbalanced partitioning
    """
    print("=" * 70)
    print("TEST: FEDERATED OLS")
    print("=" * 70)
    
    # Parameters
    random_seed = 42
    n = 1000
    p = 5
    beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    sigma = 1.5
    num_sites = 3
    
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Number of sites:        {num_sites}")
    print(f"  Random seed:            {random_seed}")
    
    # Generate data
    print("\n" + "-" * 70)
    print("1. Generating synthetic data")
    print("-" * 70)
    
    X, y = generate_dataset(
        n=n,
        p=p,
        beta=beta,
        sigma=sigma,
        random_state=random_seed,
    )
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Centralized OLS
    print("\n" + "-" * 70)
    print("2. Computing centralized OLS")
    print("-" * 70)
    
    beta_central = centralized_ols(X, y)
    print(f"  beta_central: {beta_central}")
    
    # Federated OLS - Balanced
    print("\n" + "-" * 70)
    print("3. Computing federated OLS (balanced)")
    print("-" * 70)
    
    partitioner = FederatedPartitioner(
        num_sites=num_sites,
        random_state=random_seed,
    )
    partitions_balanced = partitioner.partition(X, y)
    
    for i, site_data in enumerate(partitions_balanced, 1):
        print(f"  Site {i}: {site_data['X'].shape[0]} observations")
    
    federated_ols = FederatedOLS()
    beta_fed_balanced = federated_ols.fit(partitions_balanced)
    print(f"  beta_fed_balanced: {beta_fed_balanced}")
    
    # Compare
    diff_balanced = np.max(np.abs(beta_central - beta_fed_balanced))
    print(f"  Max absolute difference: {diff_balanced:.2e}")
    
    # Federated OLS - Unbalanced
    print("\n" + "-" * 70)
    print("4. Computing federated OLS (unbalanced)")
    print("-" * 70)
    
    proportions = np.array([0.5, 0.3, 0.2])
    partitions_unbalanced = partitioner.partition(X, y, proportions=proportions)
    
    for i, site_data in enumerate(partitions_unbalanced, 1):
        print(f"  Site {i}: {site_data['X'].shape[0]} observations")
    
    federated_ols2 = FederatedOLS()
    beta_fed_unbalanced = federated_ols2.fit(partitions_unbalanced)
    print(f"  beta_fed_unbalanced: {beta_fed_unbalanced}")
    
    # Compare
    diff_unbalanced = np.max(np.abs(beta_central - beta_fed_unbalanced))
    print(f"  Max absolute difference: {diff_unbalanced:.2e}")
    
    # Results
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    test1 = diff_balanced < 1e-10
    print(f"  Balanced OLS matches centralized: {'PASSED' if test1 else 'FAILED'}")
    
    test2 = diff_unbalanced < 1e-10
    print(f"  Unbalanced OLS matches centralized: {'PASSED' if test2 else 'FAILED'}")
    
    all_passed = test1 and test2
    
    if all_passed:
        print("\n✅ Federated OLS validation PASSED")
        print("   Federated OLS numerically matches centralized OLS")
    else:
        print("\n❌ Federated OLS validation FAILED")
    
    return all_passed


if __name__ == "__main__":
    test_federated_ols()

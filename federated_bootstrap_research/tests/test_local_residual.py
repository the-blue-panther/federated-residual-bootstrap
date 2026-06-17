"""Test for local residual bootstrap implementation."""

import numpy as np
from federated_bootstrap_research.bootstrap_methods import LocalResidualBootstrap
from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.federated.partition import FederatedPartitioner


def test_local_residual_bootstrap():
    """
    Test local residual bootstrap.
    
    Verifies:
    1. beta_hat shape correct
    2. bootstrap_betas shape correct
    3. bootstrap standard errors positive
    4. confidence interval dimensions correct
    5. no NaN values
    6. bootstrap variability exists
    """
    print("=" * 70)
    print("TEST: LOCAL RESIDUAL BOOTSTRAP")
    print("=" * 70)
    
    # Parameters
    random_seed = 42
    n = 1000
    p = 5
    beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    sigma = 1.5
    num_sites = 3
    B = 500
    confidence_level = 0.95
    
    print(f"\nConfiguration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Number of sites:        {num_sites}")
    print(f"  Bootstrap iterations:   {B}")
    print(f"  Confidence level:       {confidence_level}")
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
    
    # Partition data
    print("\n" + "-" * 70)
    print("2. Partitioning data into federated sites")
    print("-" * 70)
    
    partitioner = FederatedPartitioner(
        num_sites=num_sites,
        random_state=random_seed,
    )
    partitions = partitioner.partition(X, y)
    
    for i, site_data in enumerate(partitions, 1):
        print(f"  Site {i}: {site_data['X'].shape[0]} observations")
    
    # Fit local residual bootstrap
    print("\n" + "-" * 70)
    print("3. Running local residual bootstrap")
    print("-" * 70)
    
    bootstrap = LocalResidualBootstrap(
        n_bootstrap=B,
        confidence_level=confidence_level,
        random_state=random_seed,
    )
    results = bootstrap.fit(partitions)
    
    beta_hat = results["beta_hat"]
    bootstrap_betas = results["bootstrap_betas"]
    bootstrap_se = results["bootstrap_se"]
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    
    print(f"\n  beta_hat: {beta_hat}")
    print(f"  bootstrap_se: {bootstrap_se}")
    print(f"  Confidence intervals (95%):")
    for j in range(p):
        print(f"    beta_{j}: [{ci_lower[j]:.4f}, {ci_upper[j]:.4f}]")
    
    # Validation tests
    print("\n" + "-" * 70)
    print("4. Running validation tests")
    print("-" * 70)
    
    # Test 1: beta_hat shape correct
    test1 = beta_hat.shape == (p,)
    print(f"  Test 1 - beta_hat shape: {'PASSED' if test1 else 'FAILED'}")
    
    # Test 2: bootstrap_betas shape correct
    test2 = bootstrap_betas.shape == (B, p)
    print(f"  Test 2 - bootstrap_betas shape: {'PASSED' if test2 else 'FAILED'}")
    
    # Test 3: bootstrap standard errors positive
    test3 = np.all(bootstrap_se > 0)
    print(f"  Test 3 - bootstrap standard errors positive: {'PASSED' if test3 else 'FAILED'}")
    
    # Test 4: confidence interval dimensions correct
    test4 = ci_lower.shape == (p,) and ci_upper.shape == (p,)
    print(f"  Test 4 - CI dimensions: {'PASSED' if test4 else 'FAILED'}")
    
    # Test 5: no NaN values
    test5a = not np.any(np.isnan(beta_hat))
    test5b = not np.any(np.isnan(bootstrap_betas))
    test5c = not np.any(np.isnan(bootstrap_se))
    test5d = not np.any(np.isnan(ci_lower))
    test5e = not np.any(np.isnan(ci_upper))
    test5 = test5a and test5b and test5c and test5d and test5e
    print(f"  Test 5 - No NaN values: {'PASSED' if test5 else 'FAILED'}")
    
    # Test 6: bootstrap variability exists
    variance = np.var(bootstrap_betas, axis=0, ddof=1)
    test6 = np.all(variance > 0)
    print(f"  Test 6 - Bootstrap variability exists: {'PASSED' if test6 else 'FAILED'}")
    if test6:
        print(f"    Variance of bootstrap betas: {variance}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    all_passed = test1 and test2 and test3 and test4 and test5 and test6
    
    if all_passed:
        print("\n✅ Local residual bootstrap validation PASSED")
    else:
        print("\n❌ Local residual bootstrap validation FAILED")
    
    return all_passed


if __name__ == "__main__":
    test_local_residual_bootstrap()

"""Test suite for centralized residual bootstrap implementation."""

import numpy as np
from federated_bootstrap_research.data_generation import generate_dataset
from federated_bootstrap_research.bootstrap_methods import (
    CentralizedResidualBootstrap,
)


def test_centralized_bootstrap():
    """
    Comprehensive test for centralized residual bootstrap.
    
    Tests:
    1. beta_hat shape correct
    2. bootstrap_betas shape correct
    3. bootstrap standard errors positive
    4. confidence interval dimensions correct
    5. no NaN values anywhere
    6. bootstrap variability exists
    """
    print("=" * 70)
    print("PHASE 1: CENTRALIZED RESIDUAL BOOTSTRAP TEST")
    print("=" * 70)
    
    # Generate synthetic data using Phase 0 generator
    random_seed = 42
    n = 1000
    p = 5
    beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    sigma = 1.5
    B = 500
    confidence_level = 0.95
    
    print(f"\nData Configuration:")
    print(f"  Sample size (n):        {n}")
    print(f"  Feature dimension (p):  {p}")
    print(f"  True beta:              {beta}")
    print(f"  Error sigma:            {sigma}")
    print(f"  Bootstrap iterations:   {B}")
    print(f"  Confidence level:       {confidence_level}")
    print(f"  Random seed:            {random_seed}")
    
    # Generate dataset
    print("\n" + "-" * 70)
    print("1. Generating synthetic dataset")
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
    
    # Fit centralized bootstrap
    print("\n" + "-" * 70)
    print("2. Running centralized residual bootstrap")
    print("-" * 70)
    
    bootstrap = CentralizedResidualBootstrap(
        n_bootstrap=B,
        confidence_level=confidence_level,
        random_state=random_seed,
    )
    
    results = bootstrap.fit(X, y)
    
    beta_hat = results["beta_hat"]
    bootstrap_betas = results["bootstrap_betas"]
    bootstrap_se = results["bootstrap_se"]
    ci_lower = results["ci_lower"]
    ci_upper = results["ci_upper"]
    
    print(f"\n  Original beta_hat: {beta_hat}")
    print(f"  Bootstrap standard errors: {bootstrap_se}")
    print(f"  Confidence intervals (95%):")
    for j in range(p):
        print(f"    beta_{j}: [{ci_lower[j]:.4f}, {ci_upper[j]:.4f}]")
    
    # Test 1: beta_hat shape correct
    print("\n" + "-" * 70)
    print("3. Running validation tests")
    print("-" * 70)
    
    test1 = beta_hat.shape == (p,)
    print(f"  Test 1 - beta_hat shape: {'PASSED' if test1 else 'FAILED'} (expected ({p},), got {beta_hat.shape})")
    
    # Test 2: bootstrap_betas shape correct
    test2 = bootstrap_betas.shape == (B, p)
    print(f"  Test 2 - bootstrap_betas shape: {'PASSED' if test2 else 'FAILED'} (expected ({B}, {p}), got {bootstrap_betas.shape})")
    
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
    if not test5:
        if np.any(np.isnan(beta_hat)):
            print("    - NaN found in beta_hat")
        if np.any(np.isnan(bootstrap_betas)):
            print("    - NaN found in bootstrap_betas")
        if np.any(np.isnan(bootstrap_se)):
            print("    - NaN found in bootstrap_se")
        if np.any(np.isnan(ci_lower)):
            print("    - NaN found in ci_lower")
        if np.any(np.isnan(ci_upper)):
            print("    - NaN found in ci_upper")
    
    # Test 6: bootstrap variability exists
    # Check that variance of bootstrap coefficients > 0
    variance = np.var(bootstrap_betas, axis=0, ddof=1)
    test6 = np.all(variance > 0)
    print(f"  Test 6 - Bootstrap variability exists: {'PASSED' if test6 else 'FAILED'}")
    if test6:
        print(f"    Variance of bootstrap betas: {variance}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 VALIDATION COMPLETE")
    print("=" * 70)
    
    all_passed = test1 and test2 and test3 and test4 and test5 and test6
    
    if all_passed:
        print("\n✅ Phase 1 centralized bootstrap successfully validated!")
        print("   - beta_hat shape: PASSED")
        print("   - bootstrap_betas shape: PASSED")
        print("   - Standard errors positive: PASSED")
        print("   - Confidence intervals: PASSED")
        print("   - No NaN values: PASSED")
        print("   - Bootstrap variability: PASSED")
    else:
        print("\n❌ Phase 1 validation FAILED")
        print("   Please check the test output for details.")
    
    print("\n" + "=" * 70)
    print("Phase 1 ready for future federated bootstrap research.")
    print("=" * 70)
    
    # Optional diagnostic output
    print("\n" + "-" * 70)
    print("DIAGNOSTIC OUTPUT")
    print("-" * 70)
    print(f"\nbeta_hat (OLS estimates):")
    for j, val in enumerate(beta_hat):
        print(f"  beta_{j}: {val:.6f}")
    
    print(f"\nbootstrap_se (bootstrap standard errors):")
    for j, val in enumerate(bootstrap_se):
        print(f"  SE_{j}: {val:.6f}")
    
    print(f"\nConfidence intervals ({int(confidence_level*100)}%):")
    for j in range(p):
        print(f"  beta_{j}: [{ci_lower[j]:.6f}, {ci_upper[j]:.6f}]")
    
    print(f"\nTrue beta values for reference:")
    for j, val in enumerate(beta):
        print(f"  true_beta_{j}: {val:.6f}")
    
    print("\n" + "-" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_centralized_bootstrap()

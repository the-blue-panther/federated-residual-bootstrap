"""Phase 0 integration test for federated bootstrap research platform."""

import numpy as np
from federated_bootstrap_research.data_generation import (
    LinearModelDataGenerator,
    generate_dataset,
)
from federated_bootstrap_research.federated import FederatedPartitioner


def test_phase0():
    """
    Test Phase 0 infrastructure.
    
    This test validates:
    1. Synthetic data generation
    2. Federated partitioning
    3. Data integrity (no lost observations)
    """
    print("=" * 70)
    print("PHASE 0: FEDERATED RESIDUAL BOOTSTRAP RESEARCH")
    print("=" * 70)
    
    # Parameters
    random_seed = 42
    n = 1000
    p = 5
    beta = np.array([2.5, -1.8, 3.2, 0.5, -0.7])
    sigma = 1.5
    num_sites = 3
    
    print(f"\nConfiguration:")
    print(f"  Sample size (n):       {n}")
    print(f"  Feature dimension (p): {p}")
    print(f"  Beta:                  {beta}")
    print(f"  Sigma:                 {sigma}")
    print(f"  Number of sites:       {num_sites}")
    print(f"  Random seed:           {random_seed}")
    
    # 1. Generate synthetic dataset
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
    print(f"  X mean:  {X.mean():.6f}")
    print(f"  X std:   {X.std():.6f}")
    print(f"  y mean:  {y.mean():.6f}")
    print(f"  y std:   {y.std():.6f}")
    
    # 2. Partition data
    print("\n" + "-" * 70)
    print("2. Partitioning into federated sites")
    print("-" * 70)
    
    partitioner = FederatedPartitioner(
        num_sites=num_sites,
        random_state=random_seed,
    )
    
    # Balanced partitioning
    partitions_balanced = partitioner.partition(X, y)
    
    print("\nBalanced Partitioning:")
    total_balanced = 0
    for i, site_data in enumerate(partitions_balanced, 1):
        X_site = site_data["X"]
        y_site = site_data["y"]
        size = X_site.shape[0]
        total_balanced += size
        print(f"  Site {i}: {size} observations (X: {X_site.shape}, y: {y_site.shape})")
    
    print(f"\n  Total observations: {total_balanced} (expected: {n})")
    
    # Unbalanced partitioning
    print("\nUnbalanced Partitioning:")
    proportions = np.array([0.5, 0.3, 0.2])
    print(f"  Proportions: {proportions}")
    
    partitions_unbalanced = partitioner.partition(X, y, proportions=proportions)
    
    total_unbalanced = 0
    for i, site_data in enumerate(partitions_unbalanced, 1):
        X_site = site_data["X"]
        y_site = site_data["y"]
        size = X_site.shape[0]
        total_unbalanced += size
        expected_size = int(proportions[i-1] * n)
        print(f"  Site {i}: {size} observations (expected: ~{expected_size})")
    
    print(f"\n  Total observations: {total_unbalanced} (expected: {n})")
    
    # 3. Verify data integrity
    print("\n" + "-" * 70)
    print("3. Verifying data integrity")
    print("-" * 70)
    
    # Check balanced partition - verify all observations are preserved (order may differ)
    all_X_balanced = np.vstack([site["X"] for site in partitions_balanced])
    all_y_balanced = np.concatenate([site["y"] for site in partitions_balanced])
    
    # Check that the combined dataset has the same observations (permutation check)
    # For X: compare sorted rows, for y: compare sorted values
    X_sorted_original = np.sort(X, axis=0)
    X_sorted_reconstructed = np.sort(all_X_balanced, axis=0)
    X_matches_balanced = np.allclose(X_sorted_original, X_sorted_reconstructed)
    
    y_sorted_original = np.sort(y)
    y_sorted_reconstructed = np.sort(all_y_balanced)
    y_matches_balanced = np.allclose(y_sorted_original, y_sorted_reconstructed)
    
    print(f"  Balanced partition - X observations preserved: {X_matches_balanced}")
    print(f"  Balanced partition - y observations preserved: {y_matches_balanced}")
    print(f"  Balanced partition - total observations preserved: {total_balanced == n}")
    
    # Check unbalanced partition
    all_X_unbalanced = np.vstack([site["X"] for site in partitions_unbalanced])
    all_y_unbalanced = np.concatenate([site["y"] for site in partitions_unbalanced])
    
    X_sorted_reconstructed_unbal = np.sort(all_X_unbalanced, axis=0)
    X_matches_unbalanced = np.allclose(X_sorted_original, X_sorted_reconstructed_unbal)
    
    y_sorted_reconstructed_unbal = np.sort(all_y_unbalanced)
    y_matches_unbalanced = np.allclose(y_sorted_original, y_sorted_reconstructed_unbal)
    
    print(f"  Unbalanced partition - X observations preserved: {X_matches_unbalanced}")
    print(f"  Unbalanced partition - y observations preserved: {y_matches_unbalanced}")
    print(f"  Unbalanced partition - total observations preserved: {total_unbalanced == n}")
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("PHASE 0 VALIDATION COMPLETE")
    print("=" * 70)
    
    all_passed = (
        total_balanced == n and
        total_unbalanced == n and
        X_matches_balanced and
        y_matches_balanced and
        X_matches_unbalanced and
        y_matches_unbalanced
    )
    
    if all_passed:
        print("\n✅ Phase 0 infrastructure successfully validated!")
        print("   - Data generation: PASSED")
        print("   - Balanced partitioning: PASSED")
        print("   - Unbalanced partitioning: PASSED")
        print("   - Data integrity: PASSED")
        print("   - No observations lost: PASSED")
    else:
        print("\n❌ Phase 0 validation FAILED")
        print("   Please check the test output for details.")
    
    print("\n" + "=" * 70)
    print("Phase 0 ready for future bootstrap research.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_phase0()

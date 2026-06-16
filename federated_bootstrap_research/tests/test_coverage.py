"""Test coverage metrics."""

import numpy as np
from federated_bootstrap_research.metrics import compute_coverage


def test_coverage_function():
    """Test coverage computation."""
    print("Testing coverage metrics...")
    
    # Test 1: Single confidence interval
    ci_lower = np.array([0.8, 1.2, 2.5])
    ci_upper = np.array([1.2, 1.8, 3.5])
    true_beta = np.array([1.0, 1.5, 3.0])
    
    coverage = compute_coverage(ci_lower, ci_upper, true_beta)
    expected = np.array([1.0, 1.0, 1.0])
    
    assert np.allclose(coverage, expected)
    print("  Test 1 - Single CI: PASSED")
    
    # Test 2: Multiple simulations
    n_sim = 10
    p = 3
    ci_lower_all = np.random.randn(n_sim, p)
    ci_upper_all = ci_lower_all + np.random.uniform(0.5, 1.5, (n_sim, p))
    true_beta = np.zeros(p)
    
    coverage = compute_coverage(ci_lower_all, ci_upper_all, true_beta)
    assert coverage.shape == (p,)
    assert np.all((coverage >= 0) & (coverage <= 1))
    print("  Test 2 - Multiple simulations: PASSED")
    
    print("All coverage tests passed!")


if __name__ == "__main__":
    test_coverage_function()

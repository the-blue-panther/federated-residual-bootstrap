"""Test theoretical SE computation."""

import numpy as np
from federated_bootstrap_research.metrics import compute_theoretical_se


def test_theoretical_se():
    """Test theoretical standard error computation."""
    print("Testing theoretical SE computation...")
    
    # Test 1: Simple case with orthogonal X
    n = 100
    p = 3
    X = np.random.randn(n, p)
    sigma = 1.0
    
    se = compute_theoretical_se(X, sigma)
    
    assert se.shape == (p,)
    assert np.all(se > 0)
    print("  Test 1 - Shape and positivity: PASSED")
    
    # Test 2: Scaling with sigma
    sigma2 = 2.0
    se2 = compute_theoretical_se(X, sigma2)
    
    # SE should scale linearly with sigma
    assert np.allclose(se2 / se, sigma2 / sigma)
    print("  Test 2 - Sigma scaling: PASSED")
    
    # Test 3: Increasing n should decrease SE
    X_large = np.random.randn(1000, p)
    se_large = compute_theoretical_se(X_large, sigma)
    
    assert np.mean(se_large) < np.mean(se)
    print("  Test 3 - Sample size scaling: PASSED")
    
    print("All theoretical SE tests passed!")


if __name__ == "__main__":
    test_theoretical_se()

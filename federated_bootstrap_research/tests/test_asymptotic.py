"""Test asymptotic metrics."""

import numpy as np
from federated_bootstrap_research.metrics import compute_bias, compute_mse


def test_asymptotic_metrics():
    """Test bias and MSE computation."""
    print("Testing asymptotic metrics...")
    
    # Test 1: Bias
    estimates = np.array([[1.1, 2.1], [0.9, 1.9], [1.0, 2.0]])
    true_beta = np.array([1.0, 2.0])
    
    bias = compute_bias(estimates, true_beta)
    expected_bias = np.array([0.0, 0.0])
    
    assert np.allclose(bias, expected_bias, atol=1e-6)
    print("  Test 1 - Bias: PASSED")
    
    # Test 2: MSE
    mse = compute_mse(estimates, true_beta)
    expected_mse = np.array([0.00666667, 0.00666667])
    
    assert np.allclose(mse, expected_mse, atol=1e-6)
    print("  Test 2 - MSE: PASSED")
    
    # Test 3: Bias with non-zero bias
    estimates_biased = np.array([[1.2, 2.2], [1.1, 2.1], [1.3, 2.3]])
    bias_biased = compute_bias(estimates_biased, true_beta)
    expected_biased = np.array([0.2, 0.2])
    
    assert np.allclose(bias_biased, expected_biased, atol=1e-6)
    print("  Test 3 - Non-zero bias: PASSED")
    
    # Test 4: MSE increases with variance
    estimates_high_var = np.array([[1.5, 2.5], [0.5, 1.5], [1.0, 2.0]])
    mse_high_var = compute_mse(estimates_high_var, true_beta)
    
    assert np.all(mse_high_var > mse)
    print("  Test 4 - Variance scaling: PASSED")
    
    print("All asymptotic metrics tests passed!")


if __name__ == "__main__":
    test_asymptotic_metrics()

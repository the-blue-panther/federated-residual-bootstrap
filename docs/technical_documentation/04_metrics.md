# Evaluation Metrics

## 1. Overview

This document describes all metrics used to evaluate bootstrap methods in the project.

---

## 2. Coverage Probability

### File: `metrics/coverage.py`

### Definition

Coverage is the proportion of confidence intervals that contain the true parameter value.

$$\text{Coverage} = \frac{1}{MC} \sum_{i=1}^{MC} \mathbb{1}(\beta \in CI_i)$$

where:
- $MC$: Number of Monte Carlo simulations
- $CI_i$: Confidence interval from simulation $i$
- $\mathbb{1}$: Indicator function

### Expected Value

For a nominal 95% confidence interval:

$$E[\text{Coverage}] = 0.95$$

### Implementation

```python
def compute_coverage(ci_lower, ci_upper, true_beta):
    """
    ci_lower, ci_upper: (n_sim, p) arrays
    true_beta: (p,) array
    """
    covered = (ci_lower <= true_beta) & (true_beta <= ci_upper)
    return np.mean(covered, axis=0)
```

### Validation Criteria

| Range | Status |
|-------|--------|
| 0.93 - 0.97 | ✅ PASSED |
| < 0.93 or > 0.97 | ❌ FAILED |

---

## 3. Bias

### File: `metrics/bias.py`

### Definition

Bias measures the systematic deviation of an estimator from the true value.

$$\text{Bias} = E[\hat{\beta}] - \beta$$

### Implementation

```python
def compute_bias(estimates, true_beta):
    """
    estimates: (n_sim, p) array
    true_beta: (p,) array
    """
    mean_estimates = np.mean(estimates, axis=0)
    return mean_estimates - true_beta
```

### Interpretation

| Bias | Interpretation |
|------|----------------|
| ≈ 0 | Unbiased |
| > 0 | Overestimates |
| < 0 | Underestimates |

---

## 4. Mean Squared Error (MSE)

### File: `metrics/mse.py`

### Definition

MSE measures the average squared difference between estimates and true values.

$$\text{MSE} = E[(\hat{\beta} - \beta)^2]$$

### Implementation

```python
def compute_mse(estimates, true_beta):
    """
    estimates: (n_sim, p) array
    true_beta: (p,) array
    """
    squared_errors = (estimates - true_beta) ** 2
    return np.mean(squared_errors, axis=0)
```

### Relationship to Bias and Variance

$$\text{MSE} = \text{Bias}^2 + \text{Variance}$$

---

## 5. Theoretical Standard Errors

### File: `metrics/theoretical_se.py`

### Definition

Theoretical standard errors under homoscedasticity:

$$\text{Var}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$$

$$SE_j = \sqrt{\text{Var}(\hat{\beta})_{jj}}$$

### Implementation

```python
def compute_theoretical_se(X, sigma):
    """
    X: (n, p) feature matrix
    sigma: error standard deviation
    """
    XTX_inv = np.linalg.inv(X.T @ X)
    var_cov = sigma ** 2 * XTX_inv
    return np.sqrt(np.diag(var_cov))
```

### Comparison with Bootstrap SE

$$\text{Relative Error} = \frac{|SE_{Bootstrap} - SE_{Theoretical}|}{SE_{Theoretical}}$$

---

## 6. Wasserstein Distance

### File: `metrics/wasserstein.py`

### Definition

Wasserstein distance measures the distance between two probability distributions.

$$W(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \int ||x - y|| \, d\gamma(x, y)$$

For 1D distributions:

$$W(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx$$

### Implementation

```python
from scipy.stats import wasserstein_distance

def compute_wasserstein(dist1, dist2):
    """
    dist1, dist2: (n_samples, p) arrays
    """
    if dist1.ndim == 1:
        return wasserstein_distance(dist1, dist2)
    else:
        p = dist1.shape[1]
        distances = np.zeros(p)
        for j in range(p):
            distances[j] = wasserstein_distance(dist1[:, j], dist2[:, j])
        return distances
```

### Interpretation

| Wasserstein | Meaning |
|-------------|---------|
| Near 0 | Distributions are similar |
| Large | Distributions differ |

---

## 7. Kolmogorov-Smirnov Distance

### File: `metrics/ks_distance.py`

### Definition

KS distance is the maximum difference between two cumulative distribution functions.

$$D_{KS} = \sup_x |F_1(x) - F_2(x)|$$

### Implementation

```python
from scipy.stats import ks_2samp

def compute_ks_distance(dist1, dist2):
    """
    dist1, dist2: (n_samples, p) arrays
    """
    if dist1.ndim == 1:
        return ks_2samp(dist1, dist2).statistic
    else:
        p = dist1.shape[1]
        distances = np.zeros(p)
        for j in range(p):
            distances[j] = ks_2samp(dist1[:, j], dist2[:, j]).statistic
        return distances
```

### Interpretation

| KS | Meaning |
|----|---------|
| 0 | Identical distributions |
| 1 | Completely different |

---

## 8. Metric Summary

| Metric | Purpose | Range | Desired |
|--------|---------|-------|---------|
| Coverage | CI validity | [0, 1] | ≈ 0.95 |
| Bias | Estimation accuracy | ℝ | ≈ 0 |
| MSE | Overall error | ≥ 0 | Small |
| Theoretical SE | Reference error | > 0 | Match bootstrap |
| Wasserstein | Distribution distance | ≥ 0 | Small |
| KS | Distribution distance | [0, 1] | Small |

---

## 9. Usage Example

```python
from federated_bootstrap_research.metrics import (
    compute_coverage,
    compute_bias,
    compute_mse,
    compute_theoretical_se,
)
from federated_bootstrap_research.metrics.wasserstein import compute_wasserstein
from federated_bootstrap_research.metrics.ks_distance import compute_ks_distance

# Compute coverage
coverage = compute_coverage(ci_lower, ci_upper, true_beta)

# Compute bias
bias = compute_bias(estimates, true_beta)

# Compute MSE
mse = compute_mse(estimates, true_beta)

# Compute theoretical SE
theoretical_se = compute_theoretical_se(X, sigma)

# Compute Wasserstein distance
wasserstein = compute_wasserstein(central_betas, local_betas)

# Compute KS distance
ks = compute_ks_distance(central_betas, local_betas)
```

---

*Last Updated: 2026-06-17*

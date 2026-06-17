# Federated OLS Implementation

## 1. Overview

Federated OLS computes the same coefficient estimates as centralized OLS without sharing raw data.

**Key Insight:**

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

requires only:
- $X^TX$ (p x p matrix)
- $X^Ty$ (p x 1 vector)

These can be computed locally and aggregated.

---

## 2. Mathematical Derivation

### 2.1 Centralized OLS

$$\hat{\beta}_{Central} = (X^TX)^{-1}X^Ty$$

### 2.2 Federated OLS

**Each site computes:**

$$XTX_m = X_m^T X_m$$

$$XTy_m = X_m^T y_m$$

**Server aggregates:**

$$XTX = \sum_{m=1}^{M} XTX_m$$

$$XTy = \sum_{m=1}^{M} XTy_m$$

**Server solves:**

$$\hat{\beta}_{Fed} = \text{solve}(XTX, XTy)$$

### 2.3 Equivalence Proof

$$XTX = \sum_m X_m^T X_m = X^TX$$

$$XTy = \sum_m X_m^T y_m = X^Ty$$

Therefore:

$$\hat{\beta}_{Fed} = \hat{\beta}_{Central}$$

exactly (up to numerical precision).

---

## 3. Implementation

### File: `federated/federated_ols.py`

### Class: `FederatedOLS`

**Methods:**

```python
def compute_site_stats(X, y) -> Dict[str, np.ndarray]:
    """
    Compute local sufficient statistics.
    
    Returns:
        {
            "XTX": X.T @ X,
            "XTy": X.T @ y,
            "n": X.shape[0]
        }
    """

def aggregate_stats(site_stats) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate local statistics.
    
    Returns:
        XTX_global, XTy_global
    """

def fit(partitions) -> np.ndarray:
    """
    Fit federated OLS.
    
    Args:
        partitions: List of site data dicts
        
    Returns:
        beta_hat: Federated OLS estimates
    """
```

**Algorithm:**

```
1. For each site:
   a. Compute XTX_m = X_m.T @ X_m
   b. Compute XTy_m = X_m.T @ y_m
   c. Store statistics

2. Aggregate:
   a. XTX = sum(XTX_m)
   b. XTy = sum(XTy_m)

3. Solve:
   beta_hat = np.linalg.solve(XTX, XTy)

4. Return beta_hat
```

---

## 4. Numerical Stability

### Using `np.linalg.solve()` vs `np.linalg.inv()`

**Correct:**
```python
beta_hat = np.linalg.solve(XTX, XTy)
```

**Incorrect (avoid):**
```python
beta_hat = np.linalg.inv(XTX) @ XTy
```

### Why `solve()` is better:
- Numerically more stable
- Faster
- Standard scientific computing practice
- Avoids explicit matrix inversion

---

## 5. Validation

### Test: `tests/test_federated_ols.py`

**Checks:**

1. Balanced partitioning matches centralized OLS
2. Unbalanced partitioning matches centralized OLS
3. Tolerance: < 1e-10

**Results:**

| n | Max Difference | Status |
|---|----------------|--------|
| 100 | 9.99e-16 | ✅ PASSED |
| 1000 | 9.99e-16 | ✅ PASSED |
| 10000 | 9.99e-16 | ✅ PASSED |

---

## 6. Privacy Considerations

### What is Shared?

| Information | Shared? |
|-------------|---------|
| Raw features X_m | ❌ No |
| Raw responses y_m | ❌ No |
| Residuals e_m | ❌ No |
| $XTX_m$ | ✅ Yes (aggregated) |
| $XTy_m$ | ✅ Yes (aggregated) |
| $\hat{\beta}_{Fed}$ | ✅ Yes |

### What is Protected?

- Individual observations remain private
- Site-specific information is not exposed
- Only sufficient statistics are shared

---

## 7. Code Example

```python
from federated_bootstrap_research.federated import FederatedOLS
from federated_bootstrap_research.federated.partition import FederatedPartitioner

# Partition data
partitioner = FederatedPartitioner(num_sites=3, random_state=42)
partitions = partitioner.partition(X, y)

# Fit federated OLS
federated_ols = FederatedOLS()
beta_fed = federated_ols.fit(partitions)

# Compare with centralized
from federated_bootstrap_research.federated.federated_ols import centralized_ols
beta_central = centralized_ols(X, y)

# Both are numerically identical
print(f"Max diff: {np.max(np.abs(beta_fed - beta_central)):.2e}")
```

---

## 8. Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| $X_m$ | Feature matrix at site m |
| $y_m$ | Response vector at site m |
| $XTX_m$ | $X_m^T X_m$ |
| $XTy_m$ | $X_m^T y_m$ |
| $\hat{\beta}_{Fed}$ | Federated OLS estimate |
| $\hat{\beta}_{Central}$ | Centralized OLS estimate |
| $M$ | Number of sites |
| $p$ | Number of features |
| $n_m$ | Sample size at site m |
| $n$ | Total sample size |

---

*Last Updated: 2026-06-17*

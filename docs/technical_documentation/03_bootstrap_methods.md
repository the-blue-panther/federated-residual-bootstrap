# Bootstrap Methods

## 1. Centralized Residual Bootstrap

### File: `bootstrap_methods/centralized.py`

### Class: `CentralizedResidualBootstrap`

### 1.1 Algorithm

**Input:**
- $X$: Feature matrix (n x p)
- $y$: Response vector (n)
- $B$: Number of bootstrap iterations
- $\alpha$: Significance level

**Algorithm:**

```
1. Fit OLS on original data:
   beta_hat = solve(X^T X, X^T y)
   y_hat = X @ beta_hat
   e = y - y_hat

2. Center residuals:
   e_centered = e - mean(e)

3. For b = 1 to B:
   a. Sample e* with replacement from e_centered
   b. Generate y* = y_hat + e*
   c. Compute beta* = solve(X^T X, X^T y*)
   d. Store beta*

4. Compute:
   bootstrap_se = std(beta* across B)
   ci_lower = percentile(beta*, alpha/2)
   ci_upper = percentile(beta*, 1 - alpha/2)

5. Return results
```

### 1.2 Mathematical Formulas

**OLS:**
$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

**Fitted values:**
$$\hat{y} = X\hat{\beta}$$

**Residuals:**
$$e = y - \hat{y}$$

**Centered residuals:**
$$\tilde{e}_i = e_i - \bar{e}$$

**Bootstrap response:**
$$y_i^* = \hat{y}_i + \tilde{e}_i^*$$

**Bootstrap estimate:**
$$\hat{\beta}^* = (X^TX)^{-1}X^Ty^*$$

**Standard errors:**
$$SE_j = \text{sd}(\hat{\beta}^*_j)$$

**Confidence intervals (percentile):**
$$CI_j = [Q_{\alpha/2}(\hat{\beta}^*_j), Q_{1-\alpha/2}(\hat{\beta}^*_j)]$$

### 1.3 Output

```python
{
    "beta_hat": np.ndarray,       # OLS estimates (p,)
    "bootstrap_betas": np.ndarray, # Bootstrap estimates (B, p)
    "bootstrap_se": np.ndarray,    # Standard errors (p,)
    "ci_lower": np.ndarray,        # Lower CI bounds (p,)
    "ci_upper": np.ndarray,        # Upper CI bounds (p,)
    "residual_mean": float,        # Mean of residuals
    "centered_residuals": np.ndarray # Centered residuals (n,)
}
```

---

## 2. Local Residual Bootstrap

### File: `bootstrap_methods/local_residual.py`

### Class: `LocalResidualBootstrap`

### 2.1 Algorithm

**Input:**
- Partitions: List of site data dicts {X_m, y_m}
- $B$: Number of bootstrap iterations
- $\alpha$: Significance level

**Algorithm:**

```
1. Fit federated OLS:
   beta_hat = FederatedOLS.fit(partitions)

2. For each site m:
   a. y_hat_m = X_m @ beta_hat
   b. e_m = y_m - y_hat_m
   c. e_centered_m = e_m - mean(e_m)
   d. Store e_centered_m and y_hat_m

3. For b = 1 to B:
   a. For each site m:
      i. Sample e*_m from e_centered_m
      ii. y*_m = y_hat_m + e*_m
   b. Run federated OLS on (X_m, y*_m)
   c. Store beta*_b

4. Compute bootstrap_se and CI

5. Return results
```

### 2.2 Key Differences from Centralized

| Aspect | Centralized | Local |
|--------|-------------|-------|
| Residual Pool | Global | Site-specific |
| Resampling | From all residuals | From local residuals |
| OLS Refit | Centralized | Federated |
| Privacy | Raw data needed | Only statistics shared |

### 2.3 Privacy Properties

**What is NOT shared:**
- Raw features $X_m$
- Raw responses $y_m$
- Raw residuals $e_m$

**What IS shared:**
- Sufficient statistics $XTX_m$, $XTy_m$
- Bootstrap coefficient estimates $\hat{\beta}^*$

### 2.4 Output

```python
{
    "beta_hat": np.ndarray,       # Federated OLS estimates (p,)
    "bootstrap_betas": np.ndarray, # Bootstrap estimates (B, p)
    "bootstrap_se": np.ndarray,    # Standard errors (p,)
    "ci_lower": np.ndarray,        # Lower CI bounds (p,)
    "ci_upper": np.ndarray         # Upper CI bounds (p,)
}
```

---

## 3. Comparison: Centralized vs Local

### 3.1 Theoretical Relationship

**Research Question:**

$$\mathcal{L}(\hat{\beta}_{Local}^*) \overset{?}{\approx} \mathcal{L}(\hat{\beta}_{Central}^*)$$

### 3.2 Empirical Findings

**Coverage:**

| Distribution | Centralized | Local |
|--------------|-------------|-------|
| IID | 0.944 | 0.948 |
| Heavy-tailed | 0.948 | 0.924 |
| Skewed | 0.944 | 0.940 |
| Heteroscedastic | 0.932 | 0.932 |

**Wasserstein Distance (n=1000):**

| Distribution | Wasserstein |
|--------------|-------------|
| IID | 0.002301 |
| Heavy-tailed | 0.003458 |
| Skewed | 0.003761 |
| Heteroscedastic | 0.003545 |

### 3.3 Convergence

As $n \to \infty$:

$$W(\hat{\beta}_{Local}^*, \hat{\beta}_{Central}^*) \to 0$$

| n | Wasserstein |
|---|-------------|
| 100 | 0.01229 |
| 250 | 0.00779 |
| 500 | 0.00534 |
| 1000 | 0.00378 |
| 2500 | 0.00241 |
| 5000 | 0.00173 |
| 10000 | 0.00121 |

---

## 4. Implementation Details

### 4.1 Residual Centering

**Why center residuals?**

Classical residual bootstrap assumes $E[e_i] = 0$.

In finite samples, $\bar{e} \neq 0$.

Centering enforces the assumption:

$$\tilde{e}_i = e_i - \bar{e}$$

### 4.2 Random Number Generation

**Correct:**
```python
rng = np.random.default_rng(random_state)
e_star = rng.choice(residuals, size=n, replace=True)
```

**Incorrect (avoid):**
```python
np.random.seed(random_state)
e_star = np.random.choice(residuals, size=n, replace=True)
```

### 4.3 Confidence Intervals

**Percentile Method:**

$$CI = [Q_{\alpha/2}(\hat{\beta}^*), Q_{1-\alpha/2}(\hat{\beta}^*)]$$

**Implementation:**

```python
alpha = 1.0 - confidence_level
lower = np.percentile(bootstrap_betas, (alpha/2) * 100, axis=0)
upper = np.percentile(bootstrap_betas, (1 - alpha/2) * 100, axis=0)
```

---

## 5. Code Example

### Centralized Bootstrap

```python
from federated_bootstrap_research.bootstrap_methods import CentralizedResidualBootstrap

bootstrap = CentralizedResidualBootstrap(
    n_bootstrap=500,
    confidence_level=0.95,
    random_state=42
)
results = bootstrap.fit(X, y)
```

### Local Residual Bootstrap

```python
from federated_bootstrap_research.bootstrap_methods import LocalResidualBootstrap

bootstrap = LocalResidualBootstrap(
    n_bootstrap=500,
    confidence_level=0.95,
    random_state=42
)
results = bootstrap.fit(partitions)
```

---

*Last Updated: 2026-06-17*

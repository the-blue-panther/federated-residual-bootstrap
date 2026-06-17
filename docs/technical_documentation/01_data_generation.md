# Data Generation Module

## 1. Overview

The data generation module provides synthetic data generators for the linear model:

$$Y = X\beta + \varepsilon$$

Multiple error distributions and heterogeneity scenarios are supported.

---

## 2. Linear Model Data Generator

### File: `data_generation/linear_model.py`

### Class: `LinearModelDataGenerator`

**Purpose:** Generate IID data with Gaussian errors.

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `n` | int | Sample size | Required |
| `p` | int | Number of features | Required |
| `beta` | np.ndarray | Regression coefficients | Required |
| `sigma` | float | Error standard deviation | Required |
| `random_state` | int | Random seed | None |

**Methods:**

```python
def generate_features(self) -> np.ndarray:
    """Generate X ~ N(0, 1) of shape (n, p)"""
    
def generate_errors(self) -> np.ndarray:
    """Generate epsilon ~ N(0, sigma^2) of shape (n,)"""
    
def generate(self) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) where y = X @ beta + epsilon"""
```

**Algorithm:**

```
1. Generate X ~ N(0, 1) (n x p)
2. Generate epsilon ~ N(0, sigma^2) (n)
3. y = X @ beta + epsilon
4. Return X, y
```

---

## 3. Heavy-Tailed Error Generator

### File: `data_generation/heavy_tailed.py`

### Class: `HeavyTailedDataGenerator`

**Purpose:** Generate data with heavy-tailed errors using t-distribution.

**Error Distribution:**

$$\varepsilon \sim t_3$$

**Scaling:**

$$\text{Var}(t_3) = 3$$

$$\varepsilon = t_3 \times \frac{\sigma}{\sqrt{3}}$$

**Why t(3)?**
- Heavy tails (infinite variance? No, finite variance = 3)
- More extreme outliers than Gaussian
- Tests robustness of bootstrap methods

---

## 4. Skewed Error Generator

### File: `data_generation/skewed.py`

### Class: `SkewedDataGenerator`

**Purpose:** Generate data with skewed errors.

**Error Distribution:**

$$\varepsilon = (\text{Exp}(1) - 1) \times \sigma$$

**Properties:**
- Mean = 0
- Variance = $\sigma^2$
- Skewness > 0

**Why?**
- Tests robustness to asymmetric errors
- Real-world data often skewed

---

## 5. Heteroscedastic Error Generator

### File: `data_generation/heteroscedastic.py`

### Class: `HeteroscedasticDataGenerator`

**Purpose:** Generate data with variance depending on covariates.

**Error Distribution:**

$$\varepsilon_i \sim N(0, \sigma_i^2)$$

$$\sigma_i^2 = \sigma^2(1 + x_{i1}^2)$$

**Why?**
- Violates homoscedasticity assumption
- Tests robustness to variance heterogeneity
- More realistic for real data

---

## 6. Cross-Site Heterogeneity Generators

### File: `data_generation/site_heterogeneity.py`

### 6.1 Variance Heterogeneity

**Scenario A:** Different error variances per site.

| Site | Distribution | Variance |
|------|--------------|----------|
| 1 | $N(0, \sigma^2)$ | $\sigma^2$ |
| 2 | $N(0, 4\sigma^2)$ | $4\sigma^2$ |
| 3 | $N(0, 9\sigma^2)$ | $9\sigma^2$ |

### 6.2 Feature Heterogeneity

**Scenario B:** Different feature means per site.

| Site | Feature Distribution | Mean |
|------|---------------------|------|
| 1 | $N(0, 1)$ | 0 |
| 2 | $N(2, 1)$ | 2 |
| 3 | $N(-2, 1)$ | -2 |

### 6.3 Residual Shape Heterogeneity

**Scenario C:** Different error distributions per site.

| Site | Distribution |
|------|--------------|
| 1 | Gaussian |
| 2 | Heavy-tailed t(3) |
| 3 | Skewed Exponential |

### 6.4 Combined Heterogeneity

**Scenario D:** Combined variance, shape, and feature heterogeneity.

| Site | Feature Mean | Variance | Distribution |
|------|--------------|----------|--------------|
| 1 | 0 | $\sigma^2$ | Gaussian |
| 2 | 2 | $4\sigma^2$ | Heavy-tailed |
| 3 | -2 | $9\sigma^2$ | Skewed |

---

## 7. Factory Function

### File: `data_generation/generators.py`

**Function:** `generate_dataset()`

**Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `n` | int | Sample size | Required |
| `p` | int | Number of features | Required |
| `beta` | np.ndarray | Coefficients | Required |
| `sigma` | float | Error std dev | Required |
| `random_state` | int | Random seed | None |
| `distribution` | str | Error distribution | "iid" |

**Supported Distributions:**

| Distribution | Description |
|--------------|-------------|
| `"iid"` | Gaussian errors |
| `"heavy_tailed"` | t(3) errors |
| `"skewed"` | Exponential-1 errors |
| `"heteroscedastic"` | Variance depends on X |

**Example:**

```python
from federated_bootstrap_research.data_generation import generate_dataset

X, y = generate_dataset(
    n=1000,
    p=5,
    beta=np.array([2.5, -1.8, 3.2, 0.5, -0.7]),
    sigma=1.5,
    distribution="heavy_tailed",
    random_state=42
)
```

---

## 8. Random Number Generation

### Rule: Always use `np.random.default_rng()`

```python
# Correct
rng = np.random.default_rng(random_state)

# Incorrect - Never use
np.random.seed(random_state)
```

### Why?
- `default_rng` is the modern NumPy API
- Better random number generation
- More reproducible

---

*Last Updated: 2026-06-17*

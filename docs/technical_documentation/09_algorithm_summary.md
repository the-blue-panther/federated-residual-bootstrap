# Algorithm Summary

## 1. Complete Algorithm Pipeline

This document provides a step-by-step summary of the complete algorithm pipeline.

---

## 2. Data Generation

### 2.1 Linear Model

$$Y = X\beta + \varepsilon$$

where $\varepsilon \sim N(0, \sigma^2)$

### 2.2 Supported Error Distributions

| Distribution | Formula | Use Case |
|--------------|---------|----------|
| IID | $\varepsilon \sim N(0, \sigma^2)$ | Baseline |
| Heavy-tailed | $\varepsilon \sim t_3 \times \sigma/\sqrt{3}$ | Robustness |
| Skewed | $\varepsilon = (\text{Exp}(1) - 1) \times \sigma$ | Asymmetric errors |
| Heteroscedastic | $\varepsilon_i \sim N(0, \sigma^2(1+x_{i1}^2))$ | Variance heterogeneity |

### 2.3 Cross-Site Heterogeneity

| Scenario | Variation |
|----------|-----------|
| Variance | Different error variances per site |
| Feature | Different feature means per site |
| Residual Shape | Different error distributions per site |
| Combined | All of the above |

---

## 3. Federated Partitioning

### 3.1 Balanced Partitioning

Split observations evenly:

$$n_m = \lfloor n/M \rfloor$$

### 3.2 Unbalanced Partitioning

Split according to proportions:

$$n_m = p_m \times n$$

---

## 4. Federated OLS

### 4.1 Local Computation

$$XTX_m = X_m^T X_m$$

$$XTy_m = X_m^T y_m$$

### 4.2 Aggregation

$$XTX = \sum_m XTX_m$$

$$XTy = \sum_m XTy_m$$

### 4.3 Solve

$$\hat{\beta}_{Fed} = \text{solve}(XTX, XTy)$$

---

## 5. Centralized Residual Bootstrap

### 5.1 OLS

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

$$\hat{y} = X\hat{\beta}$$

$$e = y - \hat{y}$$

### 5.2 Centering

$$\tilde{e}_i = e_i - \bar{e}$$

### 5.3 Bootstrap Loop

For $b = 1$ to $B$:

1. Sample $e^\*$ with replacement from $\tilde{e}$
2. $y^\* = \hat{y} + e^\*$
3. $\hat{\beta}^\* = (X^TX)^{-1}X^Ty^\*$
4. Store $\hat{\beta}^\*$

### 5.4 Output

$$\text{SE} = \text{sd}(\hat{\beta}^\*)$$

$$CI = [Q_{\alpha/2}(\hat{\beta}^\*), Q_{1-\alpha/2}(\hat{\beta}^\*)]$$

---

## 6. Local Residual Bootstrap

### 6.1 Federated OLS

$$\hat{\beta}_{Fed} = \text{FederatedOLS}(\{X_m, y_m\})$$

### 6.2 Site Residuals

$$\hat{y}_m = X_m\hat{\beta}_{Fed}$$

$$e_m = y_m - \hat{y}_m$$

$$\tilde{e}_m = e_m - \bar{e}_m$$

### 6.3 Bootstrap Loop

For $b = 1$ to $B$:

1. For each site $m$:
   - Sample $e^\*_m$ from $\tilde{e}_m$
   - $y^\*_m = \hat{y}_m + e^\*_m$
2. **Run federated OLS on ALL sites** $\{X_m, y^\*_m\}$
3. **Store ONE global** $\hat{\beta}^\*$ **per iteration**

### 6.4 Critical Clarification: ONE Global Distribution

**Common Misunderstanding:** Each site produces its own distribution.

**Actual Behavior:** The server aggregates all sites' bootstrap statistics in EACH iteration, producing ONE global bootstrap distribution.

```
Iteration b = 1:
  Site 1 → (X_1, y*_{1,1}) → XTX_1, XTy_1
  Site 2 → (X_2, y*_{2,1}) → XTX_2, XTy_2
  Site 3 → (X_3, y*_{3,1}) → XTX_3, XTy_3
                    ↓
         Server aggregates → XTX, XTy → beta*_1 ← ONE global

Iteration b = 2:
  Site 1 → (X_1, y*_{1,2}) → XTX_1, XTy_1
  Site 2 → (X_2, y*_{2,2}) → XTX_2, XTy_2
  Site 3 → (X_3, y*_{3,2}) → XTX_3, XTy_3
                    ↓
         Server aggregates → XTX, XTy → beta*_2 ← ONE global

... repeat B times ...

Result: ONE global distribution: {beta*_1, beta*_2, ..., beta*_B}
```

### 6.5 Output

$$\text{SE}_{Local} = \text{sd}(\hat{\beta}^\*)$$

$$CI_{Local} = [Q_{\alpha/2}(\hat{\beta}^\*), Q_{1-\alpha/2}(\hat{\beta}^\*)]$$

**Both centralized and local bootstrap produce ONE global distribution of shape (B, p).**

---

## 7. Evaluation Metrics

### 7.1 Coverage

$$\text{Coverage} = \frac{1}{MC} \sum_{i=1}^{MC} \mathbb{1}(\beta \in CI_i)$$

### 7.2 Bias

$$\text{Bias} = E[\hat{\beta}] - \beta$$

### 7.3 MSE

$$\text{MSE} = E[(\hat{\beta} - \beta)^2]$$

### 7.4 Wasserstein Distance

$$W(\hat{\beta}_{Local}^\*, \hat{\beta}_{Central}^\*)$$

### 7.5 KS Distance

$$D_{KS} = \sup_x |F_{Local}(x) - F_{Central}(x)|$$

---

## 8. Validation Criteria

| Metric | Criterion |
|--------|-----------|
| Coverage | 0.93 - 0.97 |
| SE Relative Error | < 10% |
| Wasserstein Trend | Decreasing with n |
| KS Trend | Decreasing with n |
| MSE Trend | Decreasing with n |

---

## 9. Complete Algorithm Flowchart

```
Data Generation
    ↓
Federated Partitioning
    ↓
Federated OLS (beta_hat)
    ↓
Residual Computation (per site)
    ↓
Residual Centering (per site)
    ↓
Bootstrap Loop (B iterations)
    ↓
Distribution Comparison
    ↓
Metrics Computation
    ↓
Validation
```

---

## 10. Key Mathematical Results

### 10.1 Convergence

$$W(\hat{\beta}_{Local}^\*, \hat{\beta}_{Central}^\*) \to 0$$

as $n \to \infty$

### 10.2 Coverage

$$\text{Coverage}_{Local} \approx 0.95$$

### 10.3 Robustness

$$\text{Coverage}_{Local} \approx \text{Coverage}_{Central}$$

under all tested conditions.

---

## 11. Implementation Checklist

| Component | File |
|-----------|------|
| Data Generation | `data_generation/` |
| Federated OLS | `federated/federated_ols.py` |
| Partitioning | `federated/partition.py` |
| Centralized Bootstrap | `bootstrap_methods/centralized.py` |
| Local Bootstrap | `bootstrap_methods/local_residual.py` |
| Metrics | `metrics/` |
| Experiments | `experiments/` |
| Visualization | `visualization/` |
| Tests | `tests/` |

---

## 12. Mathematical Notation Reference

| Symbol | Description |
|--------|-------------|
| $n$ | Sample size |
| $p$ | Number of features |
| $M$ | Number of sites |
| $B$ | Number of bootstrap iterations |
| $MC$ | Number of Monte Carlo simulations |
| $\alpha$ | Significance level |
| $\beta$ | True coefficients |
| $\hat{\beta}$ | Estimated coefficients |
| $\hat{\beta}^\*$ | Bootstrap coefficients |
| $X$ | Feature matrix |
| $y$ | Response vector |
| $e$ | Residuals |
| $\tilde{e}$ | Centered residuals |
| $\sigma$ | Error standard deviation |
| $SE$ | Standard error |
| $CI$ | Confidence interval |
| $W$ | Wasserstein distance |
| $D_{KS}$ | KS distance |
| $\mathcal{L}$ | Sampling distribution |

---

*Last Updated: 2026-06-17*

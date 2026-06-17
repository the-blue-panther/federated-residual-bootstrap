# Phase 2.25 Validation Report
## Statistical Validation of Local Residual Bootstrap

**Date:** 2026-06-17

---

## 1. Executive Summary

This report presents a rigorous statistical validation of the Local Residual Bootstrap method for federated learning. The validation suite includes:

- High-power asymptotic study (n_mc=100, 7 sample sizes)
- Site imbalance stress tests (3 scenarios)
- Site count study (M=2 to 100 sites)
- Distribution robustness study (6 error distributions)

**Key Finding:** The Local Residual Bootstrap demonstrates strong empirical evidence of convergence to the centralized residual bootstrap distribution. Wasserstein distance decreases monotonically with sample size, from 0.0123 (n=100) to 0.0012 (n=10000). Coverage remains stable across all tested conditions (0.92-0.97).

---

## 2. Experimental Setup

### 2.1 Data Generating Process

$$Y = X\beta + \varepsilon$$

- $X \sim N(0, 1)$
- $\beta = [2.5, -1.8, 3.2, 0.5, -0.7]$
- $\sigma = 1.5$
- $p = 5$ features

### 2.2 Federated Configuration

- Default: 3 sites, balanced partitioning
- Site imbalance scenarios: balanced, moderately imbalanced, severely imbalanced
- Site counts: 2, 5, 10, 20, 50, 100

### 2.3 Bootstrap Configuration

- $B = 500$ bootstrap iterations
- $\alpha = 0.05$ (95% confidence intervals)
- Percentile confidence intervals
- Centered residuals

### 2.4 Evaluation Metrics

- **Coverage**: Proportion of confidence intervals containing true $\beta$
- **Wasserstein Distance**: $W(\hat{\beta}_{Fed}^*, \hat{\beta}_{Central}^*)$
- **KS Distance**: Kolmogorov-Smirnov distance between distributions
- **Bias**: $E[\hat{\beta}] - \beta$
- **MSE**: $E[(\hat{\beta} - \beta)^2]$

---

## 3. High-Power Asymptotic Study

### 3.1 Setup

- $n = [100, 250, 500, 1000, 2500, 5000, 10000]$
- $n_{MC} = 100$ per sample size
- $B = 500$
- Distribution: IID Gaussian

### 3.2 Results

| n | Central Cov | Local Cov | Wasserstein | KS |
|---|-------------|-----------|-------------|-----|
| 100 | 0.924 | 0.922 | 0.012292 | 0.054700 |
| 250 | 0.946 | 0.946 | 0.007791 | 0.053796 |
| 500 | 0.946 | 0.944 | 0.005344 | 0.053624 |
| 1000 | 0.946 | 0.948 | 0.003781 | 0.052400 |
| 2500 | 0.956 | 0.952 | 0.002412 | 0.053652 |
| 5000 | 0.946 | 0.952 | 0.001730 | 0.054532 |
| 10000 | 0.946 | 0.938 | 0.001206 | 0.053804 |

### 3.3 Interpretation

1. **Coverage**: Both centralized and local bootstrap maintain coverage near the nominal 95% level across all sample sizes.

2. **Wasserstein Distance**: Monotonically decreases from 0.0123 to 0.0012 as $n$ increases from 100 to 10000. This provides strong empirical evidence for convergence:
   $$\mathcal{L}(\hat{\beta}_{Fed}^*) \to \mathcal{L}(\hat{\beta}_{Central}^*)$$

3. **KS Distance**: Remains relatively stable around 0.05-0.055, indicating consistent distributional similarity.

**Validation:** Wasserstein decreasing: ✅ PASSED

---

## 4. Site Imbalance Study

### 4.1 Setup

- $n = 2000$ total observations
- 5 sites
- Scenarios:
  - **Balanced**: [400, 400, 400, 400, 400]
  - **Moderately Imbalanced**: [100, 200, 400, 500, 800]
  - **Severely Imbalanced**: [50, 100, 150, 300, 1400]
- $n_{MC} = 50$

### 4.2 Results

| Scenario | Coverage | Mean Bias | Mean MSE |
|----------|----------|-----------|----------|
| Balanced | 0.944 | 0.002362 | 0.001178 |
| Moderately Imbalanced | 0.948 | -0.000088 | 0.001147 |
| Severely Imbalanced | 0.948 | -0.001988 | 0.001168 |

### 4.3 Interpretation

1. **Coverage**: Remains stable (0.944-0.948) across all imbalance levels, close to the nominal 95% level.

2. **Bias**: Very small in all scenarios ($< 0.003$), indicating that site imbalance does not introduce substantial bias.

3. **MSE**: Consistently low ($\approx 0.0012$), showing that estimation quality is preserved even under severe imbalance.

**Research Question:** Does local residual bootstrap become less accurate as site imbalance increases?

**Answer:** No. Coverage, bias, and MSE remain remarkably stable across all imbalance scenarios. The method is robust to site imbalance.

---

## 5. Site Count Study

### 5.1 Setup

- $n = 10000$ total observations
- $M = [2, 5, 10, 20, 50, 100]$ sites
- $n_{MC} = 30$

### 5.2 Results

| M | Coverage | Mean Bias | Mean MSE |
|---|----------|-----------|----------|
| 2 | 0.960 | -0.000740 | 0.000246 |
| 5 | 0.940 | -0.001691 | 0.000243 |
| 10 | 0.940 | -0.000957 | 0.000235 |
| 20 | 0.940 | 0.001809 | 0.000221 |
| 50 | 0.967 | 0.001124 | 0.000194 |
| 100 | 0.967 | -0.000205 | 0.000212 |

### 5.3 Interpretation

1. **Coverage**: Remains stable (0.94-0.967) even as the number of sites increases to 100. No degradation is observed with increasing federation.

2. **MSE**: Tends to decrease slightly as $M$ increases, suggesting that finer partitioning does not harm estimation quality.

3. **Bias**: Remains consistently small ($< 0.002$) across all site counts.

**Research Question:** Does increasing federation itself degrade the approximation?

**Answer:** No. Coverage, bias, and MSE remain stable or improve slightly with more sites. The method is robust to the number of federated sites.

---

## 6. Distribution Robustness Study

### 6.1 Setup

- $n = 1000$
- 6 error distributions:
  - **IID**: $N(0, \sigma^2)$
  - **Heavy-tailed (t3)**: $t_3$ scaled
  - **Heavy-tailed (t5)**: $t_5$ scaled
  - **Skewed**: $\text{Exp}(1)-1$ scaled
  - **Gamma**: $\text{Gamma}(2,1)$ scaled and centered
  - **Heteroscedastic**: $\sigma_i^2 = 1 + x_{i1}^2$
- $n_{MC} = 30$

### 6.2 Results

| Distribution | Central Cov | Local Cov | Wasserstein | KS |
|--------------|-------------|-----------|-------------|-----|
| IID | 0.933 | 0.953 | 0.003728 | 0.053827 |
| Heavy-tailed (t3) | 0.960 | 0.967 | 0.003921 | 0.056573 |
| Heavy-tailed (t5) | 0.927 | 0.927 | 0.003834 | 0.052653 |
| Skewed | 0.933 | 0.947 | 0.003799 | 0.053400 |
| Gamma | 0.940 | 0.933 | 0.003815 | 0.053880 |
| Heteroscedastic | 0.940 | 0.940 | 0.005496 | 0.053867 |

### 6.3 Interpretation

1. **Coverage**: Remains close to 95% for all distributions (0.927-0.967).

2. **Wasserstein Distance**: Very small (0.0037-0.0055) across all distributions, indicating that the local bootstrap distribution closely matches the centralized distribution regardless of error distribution.

3. **KS Distance**: Consistently around 0.05-0.057, showing distributional similarity.

**Research Question:** Under which data-generating mechanisms does Local Residual Bootstrap fail?

**Answer:** The method does not show catastrophic failure under any tested distribution. Performance is remarkably stable across IID, heavy-tailed, skewed, Gamma, and heteroscedastic errors. The largest Wasserstein distance (0.0055) occurs under heteroscedasticity, which is expected given the violation of homoscedasticity.

---

## 7. Overall Assessment

### 7.1 Evidence Summary

| Condition | Coverage | Wasserstein Trend | Robustness |
|-----------|----------|-------------------|------------|
| Asymptotic (n → ∞) | ✅ Near 95% | ✅ Decreasing | ✅ |
| Site Imbalance | ✅ Stable | N/A | ✅ |
| Site Count (M → ∞) | ✅ Stable | N/A | ✅ |
| Heavy-tailed Errors | ✅ Near 95% | ✅ Small | ✅ |
| Skewed Errors | ✅ Near 95% | ✅ Small | ✅ |
| Heteroscedastic Errors | ✅ Near 95% | ✅ Small | ✅ |

### 7.2 Key Findings

1. **Convergence**: Wasserstein distance decreases monotonically with sample size, providing strong empirical evidence that:
   $$\mathcal{L}(\hat{\beta}_{Fed}^*) \to \mathcal{L}(\hat{\beta}_{Central}^*)$$

2. **Robustness**: Coverage remains near nominal levels across all tested conditions:
   - Sample sizes: 100 to 10000
   - Site imbalance: balanced to severe
   - Number of sites: 2 to 100
   - Error distributions: IID, heavy-tailed, skewed, Gamma, heteroscedastic

3. **Stability**: Bias and MSE remain consistently small across all scenarios.

### 7.3 Final Assessment

**Is Local Residual Bootstrap sufficiently promising to justify further research?**

**Answer: Yes.**

The empirical evidence strongly supports the viability of Local Residual Bootstrap as a federated analogue of the centralized residual bootstrap. The method:

- ✅ Converges to the centralized distribution as $n \to \infty$
- ✅ Maintains correct coverage under nominal conditions
- ✅ Is robust to site imbalance
- ✅ Scales well with number of sites
- ✅ Performs reliably under non-Gaussian errors

These findings justify proceeding to Phase 3, where more sophisticated federated bootstrap methods (Residual Summary Bootstrap, Multiplier Bootstrap, Gradient Bootstrap, ReBoot) can be developed and compared against this promising baseline.

---

## 8. Recommendations for Phase 3

1. **Priority**: Implement Residual Summary Bootstrap as the next candidate method
2. **Comparison**: Benchmark against Local Residual Bootstrap
3. **Evaluation**: Use the same metrics (coverage, Wasserstein, KS, bias, MSE)
4. **Stress Testing**: Apply the same stress test suite (asymptotic, site imbalance, site count, distribution robustness)

---

*Report generated: 2026-06-17*

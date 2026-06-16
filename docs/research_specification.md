# Federated Residual Bootstrap Research Platform
## Ground Truth Research Document

---

# Purpose Of This Document

This document defines:

1. The scientific objective of the project.
2. The exact simulation goals.
3. The assumptions under which experiments are performed.
4. The meaning of success and failure.
5. The relationship between theory and simulation.

This document should be treated as the authoritative reference for all future code generation, experiment design, implementation decisions, and UI development.

If future generated code conflicts with this document, the document takes precedence.

---

# High-Level Research Goal

The project investigates whether a classical residual bootstrap procedure can be reproduced in a federated learning environment without sharing raw data.

Specifically, we seek a federated analogue of the centralized residual bootstrap.

The central research question is:

$$
\mathcal{L}
\left(
\hat{\beta}_{Fed}^{*}
\right)
\overset{?}{\approx}
\mathcal{L}
\left(
\hat{\beta}_{Central}^{*}
\right)
$$

where

$$
\hat{\beta}_{Fed}^{*}
$$

is a bootstrap estimator produced under federation, and

$$
\hat{\beta}_{Central}^{*}
$$

is the bootstrap estimator produced from the fully pooled centralized dataset.

---

# Why This Matters

Classical residual bootstrap requires access to:

$$
\{e_1,e_2,\ldots,e_N\}
$$

the complete residual pool.

In federated learning:

- raw observations cannot be shared
- features cannot be shared
- labels cannot be shared
- residuals may also be private

Therefore the classical residual bootstrap cannot be directly implemented.

The project investigates whether equivalent inference can still be obtained.

---

# Long-Term Scientific Objective

Develop a methodology that:

- preserves privacy
- avoids raw data transfer
- avoids residual pooling
- produces valid uncertainty quantification
- reproduces centralized inference

including:

- standard errors
- confidence intervals
- hypothesis tests

---

# Phase Structure Of Research

The project is divided into multiple phases.

---

# Phase 0

Infrastructure Construction

Goal:

Build the software framework needed for all future simulations.

This phase contains:

- dataset generation
- federated partitioning
- configuration system
- experiment runners
- metrics
- visualization
- Streamlit dashboard

No scientific conclusions should be drawn during this phase.

---

# Phase 1

Simulation Environment Construction

Goal:

Generate synthetic datasets under controlled assumptions.

We must be able to simulate:

1. IID data
2. Site heterogeneity
3. Heavy-tailed errors
4. Skewed errors
5. Site imbalance

The objective is not inference.

The objective is to create controlled worlds where bootstrap procedures can be tested.

---

# Phase 2

Centralized Bootstrap Benchmark

Goal:

Implement the classical residual bootstrap.

This serves as the gold standard.

All federated methods are compared against this benchmark.

No federated method is allowed to claim success unless compared against the centralized bootstrap.

---

# Phase 3

Federated Bootstrap Candidates

The project currently considers five candidate methods.

---

## Method 1

Local Residual Bootstrap

Each site resamples only its own residuals.

Question:

$$
\mathcal{L}
\left(
\hat{\beta}_{Fed}^{*}
\right)
\overset{?}{\to}
\mathcal{L}
\left(
\hat{\beta}_{Central}^{*}
\right)
$$

---

## Method 2

Residual Summary Bootstrap

Residuals are never shared.

Instead, summary information is shared.

Examples:

- means
- variances
- quantiles
- moments

Question:

Can a global residual distribution be reconstructed?

---

## Method 3

Multiplier Bootstrap

Generate weights

$$
w_i
$$

such that

$$
E[w_i]=0
$$

and

$$
Var(w_i)=1
$$

Construct

$$
e_i^*=w_i e_i
$$

Question:

Can multiplier bootstrap approximate residual bootstrap inference?

---

## Method 4

Gradient Bootstrap

Bootstrap gradients instead of observations.

Question:

Can score-based perturbations reproduce bootstrap variability?

---

## Method 5

ReBoot-Style Bootstrap

Generate synthetic surrogate datasets.

Question:

Can synthetic bootstrap worlds reproduce centralized uncertainty quantification?

---

# What Simulations Are Trying To Discover

The simulations are not merely generating numbers.

They are attempting to answer scientific questions.

---

## Question A

Does local residual resampling work?

---

## Question B

If it works, under what assumptions?

---

## Question C

If it fails, where does it fail?

Examples:

- heterogeneity
- imbalance
- skewness
- heavy tails

---

## Question D

Does asymptotic convergence occur?

As

$$
n \rightarrow \infty
$$

does the federated distribution approach the centralized distribution?

---

# The Role Of Asymptotic Studies

A major objective of the project is to investigate empirical asymptotic behaviour.

Theoretical conjectures should not be accepted without simulation evidence.

For increasing values of

$$
n
$$

such as

$$
100,
500,
1000,
5000,
10000,
20000
$$

we will study:

- bias
- variance
- coverage
- distributional distance

The goal is to observe whether convergence patterns emerge.

---

# Evaluation Metrics

Every method must be evaluated using the same metrics.

---

## Coverage Probability

Estimate:

$$
P(\beta \in CI)
$$

Desired value:

$$
0.95
$$

for a nominal 95% confidence interval.

---

## Bias

Compute:

$$
Bias
=
E[\hat{\beta}]
-\beta
$$

---

## Mean Squared Error

Compute:

$$
MSE
=
E
\left[
(\hat{\beta}-\beta)^2
\right]
$$

---

## Runtime

Measure computational cost.

---

## Communication Cost

Measure communication burden.

---

## KS Distance

Compare bootstrap distributions.

---

## Wasserstein Distance

Measure distributional similarity.

---

## QQ Diagnostics

Visual comparison of distributions.

---

# Distributional Convergence Dashboard

One of the most important analyses in the project.

For increasing

$$
n
$$

compute:

$$
W
\left(
\hat{\beta}_{Fed}^{*},
\hat{\beta}_{Central}^{*}
\right)
$$

where

$$
W
$$

is the Wasserstein distance.

If convergence occurs, the distance should decrease.

This provides empirical evidence for:

$$
\mathcal L(\hat\beta^*_{Fed})
\to
\mathcal L(\hat\beta^*_{Central})
$$

---

# Site Heterogeneity Stress Tests

The project must explicitly test:

$$
\varepsilon_{im}
\sim
N
\left(
0,
\sigma_m^2
\right)
$$

where

$$
\sigma_m^2
\neq
\sigma_j^2
$$

for different sites.

Reason:

Methods often appear valid under IID assumptions but fail under heterogeneity.

---

# Site Imbalance Stress Tests

The project must explicitly test:

$$
n_m
\neq
n_j
$$

for different sites.

This reflects realistic federated healthcare settings.

---

# Meaning Of Success

A federated bootstrap method is considered promising if:

1. Coverage approaches centralized coverage.
2. Bias remains small.
3. MSE remains competitive.
4. Wasserstein distance decreases with increasing sample size.
5. Communication costs remain low.
6. Results remain stable under moderate heterogeneity.

---

# Meaning Of Failure

A federated bootstrap method is considered unsuitable if:

1. Coverage collapses.
2. Bias grows with sample size.
3. Distributional distance remains large.
4. Strong sensitivity to partitioning occurs.
5. Severe instability appears under heterogeneity.

---

# Final Objective

The simulations are not the final result.

The simulations are a scientific instrument used to determine:

Which federated bootstrap methodology deserves theoretical development and further research.

The primary purpose of the simulation platform is therefore:

To identify the most promising direction for a rigorous federated residual bootstrap methodology.
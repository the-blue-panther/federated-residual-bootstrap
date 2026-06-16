# Federated Residual Bootstrap Research Platform

## Phase 0: Foundation

This is a research simulation platform for studying Federated Residual Bootstrap methods in distributed learning environments.

## Research Goal

Compare various bootstrap procedures in federated settings:

- Centralized Residual Bootstrap
- Local Residual Bootstrap  
- Residual Summary Bootstrap
- Multiplier Bootstrap
- Gradient Bootstrap
- ReBoot-style Bootstrap

The primary objective is to evaluate:

$$\hat{\beta}^{*}_{Fed} \quad \text{vs} \quad \hat{\beta}^{*}_{Central}$$

## Phase 0 Infrastructure

Current implementation provides:

1. **Configuration Management**: YAML-based configuration (`config/default.yaml`)
2. **Synthetic Data Generation**: Linear model data generator with reproducibility
3. **Federated Partitioning**: Balanced and unbalanced data distribution across sites
4. **Reproducibility**: Seed-controlled random number generation
5. **Testing**: Validation of core infrastructure

## Project Structure

```
federated_bootstrap_research/
├── config/
│   └── default.yaml          # Configuration settings
├── data_generation/
│   ├── __init__.py
│   ├── linear_model.py       # Linear model data generator
│   └── generators.py         # Factory functions
├── federated/
│   ├── __init__.py
│   └── partition.py          # Data partitioner
├── tests/
│   ├── __init__.py
│   └── test_phase0.py        # Phase 0 validation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```python
from data_generation.generators import generate_dataset
import numpy as np

X, y = generate_dataset(
    n=1000,
    p=5,
    beta=np.array([2.5, -1.8, 3.2, 0.5, -0.7]),
    sigma=1.5,
    random_state=42
)
```

### Partition Data Federated

```python
from federated.partition import FederatedPartitioner

partitioner = FederatedPartitioner(num_sites=3, random_state=42)

# Balanced partitioning
partitions = partitioner.partition(X, y)

# Unbalanced partitioning
proportions = np.array([0.5, 0.3, 0.2])
partitions = partitioner.partition(X, y, proportions=proportions)
```

## Testing

Run the Phase 0 validation test:

```bash
python -m tests.test_phase0
```

Expected output includes:
- Dataset generation statistics
- Federated site sizes
- Data integrity verification
- Final validation status

## Requirements

- Python >= 3.11
- NumPy >= 1.24.0
- PyYAML >= 6.0
- pytest >= 7.0.0

## Next Phases

Future phases will implement:

1. Bootstrap method implementations
2. Experiment runners
3. Metric computation and comparison
4. Visualization and analysis tools
5. Simulation studies

## License

Academic Research Use

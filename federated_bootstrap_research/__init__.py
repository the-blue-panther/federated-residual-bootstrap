"""Federated Residual Bootstrap Research Platform.

A simulation platform for studying Federated Residual Bootstrap methods.
"""

__version__ = "0.1.0"

from federated_bootstrap_research.data_generation import (
    LinearModelDataGenerator,
    create_generator,
    generate_dataset,
)
from federated_bootstrap_research.federated import FederatedPartitioner

__all__ = [
    "LinearModelDataGenerator",
    "create_generator",
    "generate_dataset",
    "FederatedPartitioner",
]

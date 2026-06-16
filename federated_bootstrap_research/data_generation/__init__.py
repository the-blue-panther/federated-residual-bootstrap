"""Data generation module for synthetic linear model data."""

from federated_bootstrap_research.data_generation.linear_model import (
    LinearModelDataGenerator,
)
from federated_bootstrap_research.data_generation.generators import (
    create_generator,
    generate_dataset,
)

__all__ = [
    "LinearModelDataGenerator",
    "create_generator",
    "generate_dataset",
]

"""Federated data partitioner for distributing observations across sites."""

import numpy as np
from typing import List, Dict, Optional, Union


class FederatedPartitioner:
    """
    Partitions data into federated sites with balanced or unbalanced splits.
    
    The partitioner distributes observations (X, y) across multiple sites
    according to specified proportions.
    
    Parameters
    ----------
    num_sites : int
        Number of federated sites.
    random_state : Optional[int]
        Random seed for reproducibility (used for shuffling).
    """
    
    def __init__(
        self,
        num_sites: int,
        random_state: Optional[int] = None,
    ):
        """Initialize the federated partitioner."""
        self.num_sites = num_sites
        self.rng = np.random.default_rng(random_state)
    
    def partition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        proportions: Optional[np.ndarray] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Partition data into federated sites.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n, p).
        y : np.ndarray
            Response vector of shape (n,).
        proportions : Optional[np.ndarray]
            Proportions for each site. If None, balanced partitioning is used.
            Must sum to 1.0 and have length equal to num_sites.
        
        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of dictionaries, each containing 'X' and 'y' for a site.
        """
        n_samples = X.shape[0]
        
        # Generate indices and shuffle
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Determine partition sizes
        if proportions is None:
            # Balanced partitioning
            sizes = self._balanced_sizes(n_samples)
        else:
            # Unbalanced partitioning
            sizes = self._unbalanced_sizes(n_samples, proportions)
        
        # Split data
        partitions = []
        start_idx = 0
        for size in sizes:
            end_idx = start_idx + size
            site_indices = indices[start_idx:end_idx]
            
            partitions.append({
                "X": X[site_indices, :],
                "y": y[site_indices],
            })
            start_idx = end_idx
        
        return partitions
    
    def _balanced_sizes(self, n_samples: int) -> List[int]:
        """
        Compute balanced partition sizes.
        
        Distributes observations as evenly as possible across sites.
        
        Parameters
        ----------
        n_samples : int
            Total number of observations.
        
        Returns
        -------
        List[int]
            Number of observations for each site.
        """
        base_size = n_samples // self.num_sites
        remainder = n_samples % self.num_sites
        
        sizes = [base_size] * self.num_sites
        for i in range(remainder):
            sizes[i] += 1
        
        return sizes
    
    def _unbalanced_sizes(
        self,
        n_samples: int,
        proportions: np.ndarray,
    ) -> List[int]:
        """
        Compute unbalanced partition sizes based on proportions.
        
        Parameters
        ----------
        n_samples : int
            Total number of observations.
        proportions : np.ndarray
            Proportions for each site (must sum to 1.0).
        
        Returns
        -------
        List[int]
            Number of observations for each site.
        
        Raises
        ------
        ValueError
            If proportions do not sum to 1.0 or length does not match num_sites.
        """
        if len(proportions) != self.num_sites:
            raise ValueError(
                f"Number of proportions ({len(proportions)}) does not match "
                f"num_sites ({self.num_sites})"
            )
        
        if not np.isclose(np.sum(proportions), 1.0):
            raise ValueError(
                f"Proportions must sum to 1.0, got {np.sum(proportions)}"
            )
        
        # Compute sizes and handle rounding
        sizes = (proportions * n_samples).astype(int)
        
        # Adjust for rounding errors to ensure total matches n_samples
        diff = n_samples - np.sum(sizes)
        if diff > 0:
            # Add remaining observations to largest proportion sites
            sorted_indices = np.argsort(proportions)[::-1]
            for i in range(diff):
                sizes[sorted_indices[i % len(sorted_indices)]] += 1
        elif diff < 0:
            # Remove excess observations from smallest proportion sites
            sorted_indices = np.argsort(proportions)
            for i in range(-diff):
                sizes[sorted_indices[i % len(sorted_indices)]] -= 1
        
        # Ensure all sizes are non-negative
        if any(size < 0 for size in sizes):
            raise ValueError(
                "Partition sizes resulted in negative values. "
                "Please adjust proportions."
            )
        
        return sizes.tolist()

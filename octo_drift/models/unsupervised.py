"""
Unsupervised model (MCD - Unknown Classes Model) for octo-drift.
"""

from typing import List
import numpy as np
from ..core.structures import Example, SPFMiC
from ..core.distance import euclidean_distance


class UnsupervisedModel:
    """
    Model of Unknown Classes (MCD).

    Maintains micro-clusters discovered from novelty detection.
    These represent patterns that don't match known classes.

    Attributes:
        spfmics: List of discovered micro-clusters
    """

    def __init__(self):
        self.spfmics: List[SPFMiC] = []

    def classify(self, example: Example, k: int, timestamp: int) -> float:
        """
        Classify example using discovered unknown patterns.

        Args:
            example: Example to classify
            k: Number of clusters parameter
            timestamp: Current time

        Returns:
            Predicted label or -1 if no match
        """
        if not self.spfmics:
            return -1.0

        typicalities = []
        candidate_spfmics = []

        # Find micro-clusters within radius
        for spfmic in self.spfmics:
            distance = euclidean_distance(example, spfmic)

            if distance <= spfmic.get_radius(1.0):  # Unsupervised radius
                typicality = spfmic.calculate_typicality(example.point, k, distance)
                typicalities.append(typicality)
                candidate_spfmics.append(spfmic)

        # No match found
        if not candidate_spfmics:
            return -1.0

        # Select cluster with max typicality
        max_idx = np.argmax(typicalities)
        selected_spfmic = candidate_spfmics[max_idx]

        # Update timestamp
        selected_spfmic.updated = timestamp

        return selected_spfmic.label

    def add_spfmic(self, spfmic: SPFMiC) -> None:
        """Add a newly discovered micro-cluster."""
        self.spfmics.append(spfmic)

    def get_spfmics(self) -> List[SPFMiC]:
        """Get all micro-clusters."""
        return self.spfmics

    def remove_old_spfmics(self, threshold: int, current_time: int) -> None:
        """
        Remove obsolete micro-clusters.

        Args:
            threshold: Time window threshold
            current_time: Current timestamp
        """
        self.spfmics = [
            spfmic
            for spfmic in self.spfmics
            if not (
                current_time - spfmic.created > threshold
                and current_time - spfmic.updated > threshold
            )
        ]

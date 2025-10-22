"""
Novelty detection logic for octo-drift.
"""

from typing import List, Tuple
from collections import Counter
import numpy as np
from ..core.structures import Example, SPFMiC
from ..core.distance import euclidean_distance
from ..core.fuzzy_functions import (
    FuzzyCMeans,
    fuzzy_silhouette,
    create_spfmics_from_clusters,
)
from ..models.supervised import SupervisedModel
from ..models.unsupervised import UnsupervisedModel


class NoveltyDetector:
    """
    Detects novel classes in the data stream.

    Uses fuzzy clustering, silhouette validation, and overlap factor
    to distinguish between extensions of known classes and true novelties.

    Attributes:
        k_short: Number of clusters for unknown buffer
        phi: Overlap threshold for novelty vs known class
        min_weight: Minimum cluster size
        alpha: Pertinence exponent
        theta: Typicality exponent
        fuzzification: Fuzzification parameter
        novelty_counter: Counter for generating novelty labels
    """

    def __init__(
        self,
        k_short: int,
        phi: float,
        min_weight: int,
        alpha: float,
        theta: float,
        fuzzification: float,
    ):
        self.k_short = k_short
        self.phi = phi
        self.min_weight = min_weight
        self.alpha = alpha
        self.theta = theta
        self.fuzzification = fuzzification
        self.novelty_counter = 100.0  # Start labeling novelties from 100

    def detect(
        self,
        unknown_buffer: List[Example],
        supervised_model: SupervisedModel,
        unsupervised_model: UnsupervisedModel,
        timestamp: int,
    ) -> Tuple[List[Example], bool]:
        """
        Perform multi-class novelty detection on unknown buffer.

        Args:
            unknown_buffer: Examples classified as unknown
            supervised_model: MCC
            unsupervised_model: MCD
            timestamp: Current time

        Returns:
            Tuple of (remaining_unknown_examples, novelty_detected_flag)
        """
        if len(unknown_buffer) <= self.k_short:
            return unknown_buffer, False

        novelty_detected = False

        # Cluster unknown examples
        clusterer = FuzzyCMeans(self.k_short, self.fuzzification)
        clusterer.fit(unknown_buffer)

        # Calculate silhouette for validation
        silhouettes = fuzzy_silhouette(clusterer, unknown_buffer, self.alpha)

        # Identify valid clusters (positive silhouette + min size)
        valid_clusters = [
            i
            for i in range(len(silhouettes))
            if silhouettes[i] > 0
            and len(clusterer.get_cluster_members(i, unknown_buffer)) >= self.min_weight
        ]

        if not valid_clusters:
            return unknown_buffer, False

        # Create candidate SPFMiCs
        candidate_spfmics = create_spfmics_from_clusters(
            unknown_buffer,
            clusterer,
            label=-1,  # Temporary label
            alpha=self.alpha,
            theta=self.theta,
            min_weight=self.min_weight,
            timestamp=timestamp,
        )

        # Get known micro-clusters
        known_spfmics = supervised_model.get_spfmics()

        # Process each valid cluster
        for cluster_idx in valid_clusters:
            if cluster_idx >= len(candidate_spfmics):
                continue

            candidate = candidate_spfmics[cluster_idx]
            cluster_members = clusterer.get_cluster_members(cluster_idx, unknown_buffer)

            # Calculate overlap factors with known clusters
            overlap_factors = []
            for known in known_spfmics:
                di = known.radius_nd
                dj = candidate.radius_nd
                distance = euclidean_distance(known.centroid, candidate.centroid)

                if distance > 0:
                    overlap_factors.append((di + dj) / distance)
                else:
                    overlap_factors.append(float("inf"))

            if not overlap_factors:
                # No known clusters - treat as novelty
                self._process_as_novelty(
                    candidate,
                    cluster_members,
                    unknown_buffer,
                    unsupervised_model,
                    timestamp,
                )
                novelty_detected = True
                continue

            min_overlap = min(overlap_factors)
            min_overlap_idx = overlap_factors.index(min_overlap)

            if min_overlap <= self.phi:
                # Extension of known class
                known_label = known_spfmics[min_overlap_idx].label
                self._process_as_extension(
                    candidate,
                    cluster_members,
                    unknown_buffer,
                    unsupervised_model,
                    known_label,
                    timestamp,
                )
            else:
                # True novelty
                self._process_as_novelty(
                    candidate,
                    cluster_members,
                    unknown_buffer,
                    unsupervised_model,
                    timestamp,
                )
                novelty_detected = True

        return unknown_buffer, novelty_detected

    def _process_as_extension(self, candidate: SPFMiC, members: List[Example],
                            buffer: List[Example], mcd: UnsupervisedModel,
                            label: float, timestamp: int) -> None:
        """Process cluster as extension of known class."""
        candidate.label = label
        
        true_labels = [ex.true_label for ex in members]
        most_common_label = Counter(true_labels).most_common(1)[0][0]
        
        if most_common_label == label:
            candidate.true_label = most_common_label
            mcd.add_spfmic(candidate)
        
        # FIX: Remover por identidade, não por igualdade
        for member in members:
            member.predicted_label = candidate.label
            try:
                buffer.remove(member)
            except ValueError:
                pass

    def _process_as_novelty(self, candidate: SPFMiC, members: List[Example],
                        buffer: List[Example], mcd: UnsupervisedModel,
                        timestamp: int) -> None:
        """Process cluster as true novelty."""
        candidate.label = self.novelty_counter
        self.novelty_counter += 1
        
        true_labels = [ex.true_label for ex in members]
        most_common_label = Counter(true_labels).most_common(1)[0][0]
        candidate.true_label = most_common_label
        
        mcd.add_spfmic(candidate)
        
        # FIX: Remover por identidade, não por igualdade
        for member in members:
            member.predicted_label = candidate.label
            try:
                buffer.remove(member)
            except ValueError:
                pass

    def generate_novelty_label(self) -> float:
        """Generate new novelty label."""
        label = self.novelty_counter
        self.novelty_counter += 1
        return label

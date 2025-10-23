"""
Novelty detection logic for octo-drift.
"""

import logging
from typing import List, Tuple, Set
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

logger = logging.getLogger(__name__)


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

        logger.debug(f"[ND t={timestamp}] Buffer size: {len(unknown_buffer)}")

        try:
            # Cluster unknown examples
            clusterer = FuzzyCMeans(self.k_short, self.fuzzification)
            clusterer.fit(unknown_buffer)

            logger.debug(f"[ND t={timestamp}] FCM converged, {self.k_short} clusters")

        except Exception as e:
            logger.error(f"[ND t={timestamp}] FCM failed: {e}")
            return unknown_buffer, False

        # Calculate silhouette for validation
        silhouettes = fuzzy_silhouette(clusterer, unknown_buffer, self.alpha)

        logger.debug(
            f"[ND t={timestamp}] Silhouettes: {[f'{s:.3f}' for s in silhouettes]}"
        )

        # Identify valid clusters (positive silhouette + min size)
        valid_clusters = []
        for i in range(len(silhouettes)):
            cluster_size = len(clusterer.get_cluster_members(i, unknown_buffer))
            if silhouettes[i] > 0 and cluster_size >= self.min_weight:
                valid_clusters.append(i)
                logger.debug(
                    f"[ND t={timestamp}] Cluster {i}: valid (sil={silhouettes[i]:.3f}, size={cluster_size})"
                )

        if not valid_clusters:
            logger.debug(f"[ND t={timestamp}] No valid clusters found")
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

        logger.debug(
            f"[ND t={timestamp}] Created {len(candidate_spfmics)} candidate SPFMiCs"
        )

        # Get known micro-clusters
        known_spfmics = supervised_model.get_spfmics()

        logger.debug(f"[ND t={timestamp}] Known SPFMiCs: {len(known_spfmics)}")

        # CORREÇÃO: Usar set de IDs para rastrear exemplos processados
        processed_example_ids: Set[int] = set()

        # Process each valid cluster
        for cluster_idx in valid_clusters:
            if cluster_idx >= len(candidate_spfmics):
                continue

            candidate = candidate_spfmics[cluster_idx]
            cluster_members = clusterer.get_cluster_members(cluster_idx, unknown_buffer)

            logger.debug(
                f"[ND t={timestamp}] Processing cluster {cluster_idx} with {len(cluster_members)} members"
            )

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
                logger.info(
                    f"[ND t={timestamp}] Cluster {cluster_idx} -> NOVELTY (no known clusters)"
                )
                self._process_as_novelty(
                    candidate,
                    cluster_members,
                    processed_example_ids,
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
                logger.info(
                    f"[ND t={timestamp}] Cluster {cluster_idx} -> EXTENSION of class {known_label} "
                    f"(overlap={min_overlap:.3f})"
                )
                self._process_as_extension(
                    candidate,
                    cluster_members,
                    processed_example_ids,
                    unsupervised_model,
                    known_label,
                    timestamp,
                )
            else:
                # True novelty
                logger.info(
                    f"[ND t={timestamp}] Cluster {cluster_idx} -> NOVELTY "
                    f"(min_overlap={min_overlap:.3f} > phi={self.phi})"
                )
                self._process_as_novelty(
                    candidate,
                    cluster_members,
                    processed_example_ids,
                    unsupervised_model,
                    timestamp,
                )
                novelty_detected = True

        # CORREÇÃO: Remover exemplos processados usando IDs
        remaining_buffer = [
            ex for ex in unknown_buffer if id(ex) not in processed_example_ids
        ]

        logger.debug(
            f"[ND t={timestamp}] Buffer: {len(unknown_buffer)} -> {len(remaining_buffer)} "
            f"(removed {len(processed_example_ids)})"
        )

        return remaining_buffer, novelty_detected

    def _process_as_extension(
        self,
        candidate: SPFMiC,
        members: List[Example],
        processed_ids: Set[int],
        mcd: UnsupervisedModel,
        label: float,
        timestamp: int,
    ) -> None:
        """Process cluster as extension of known class."""
        candidate.label = label

        true_labels = [ex.true_label for ex in members]
        most_common_label = Counter(true_labels).most_common(1)[0][0]

        # Só adiciona ao MCD se a maioria dos exemplos pertence à classe atribuída
        if most_common_label == label:
            candidate.true_label = most_common_label
            mcd.add_spfmic(candidate)
            logger.debug(
                f"[ND] Added extension SPFMiC (label={label}, true={most_common_label}, n={len(members)})"
            )

        # Marcar exemplos como processados
        for member in members:
            member.predicted_label = candidate.label
            processed_ids.add(id(member))

    def _process_as_novelty(
        self,
        candidate: SPFMiC,
        members: List[Example],
        processed_ids: Set[int],
        mcd: UnsupervisedModel,
        timestamp: int,
    ) -> None:
        """Process cluster as true novelty."""
        candidate.label = self.novelty_counter
        self.novelty_counter += 1

        true_labels = [ex.true_label for ex in members]
        most_common_label = Counter(true_labels).most_common(1)[0][0]
        candidate.true_label = most_common_label

        mcd.add_spfmic(candidate)

        logger.debug(
            f"[ND] Added novelty SPFMiC (label={candidate.label}, true={most_common_label}, n={len(members)})"
        )

        # Marcar exemplos como processados
        for member in members:
            member.predicted_label = candidate.label
            processed_ids.add(id(member))

    def generate_novelty_label(self) -> float:
        """Generate new novelty label."""
        label = self.novelty_counter
        self.novelty_counter += 1
        return label

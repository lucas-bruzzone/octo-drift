"""
Core data structures for octo-drift algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class Example:
    """
    Represents a data point in the stream.

    Attributes:
        point: Feature vector (excludes class label)
        true_label: Ground truth class label
        predicted_label: Assigned label by classifier
        timestamp: Time when example arrived
    """

    point: npt.NDArray[np.float64]
    true_label: float
    predicted_label: float = -1.0
    timestamp: int = 0

    def __post_init__(self):
        """Ensure point is numpy array."""
        if not isinstance(self.point, np.ndarray):
            self.point = np.array(self.point, dtype=np.float64)

    @classmethod
    def from_array(
        cls, data: npt.NDArray[np.float64], has_label: bool = True, timestamp: int = 0
    ) -> "Example":
        """
        Create Example from array where last column is label.

        Args:
            data: Array with features and optional label
            has_label: Whether last column contains label
            timestamp: Time index

        Returns:
            Example instance
        """
        if has_label:
            point = data[:-1]
            label = data[-1]
        else:
            point = data
            label = -1.0

        return cls(point=point, true_label=label, timestamp=timestamp)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.point)


@dataclass
class SPFMiC:
    """
    Summary of Pertinence and Typicality for Micro-Cluster.

    Represents a fuzzy micro-cluster with membership and typicality information.

    Attributes:
        centroid: Cluster center
        n: Number of examples
        alpha: Pertinence exponent
        theta: Typicality exponent
        label: Assigned class label
        true_label: Most frequent true label in cluster
        created: Creation timestamp
        updated: Last update timestamp
        cf1_pertinence: Weighted sum by pertinence
        cf1_typicality: Weighted sum by typicality
        me: Sum of pertinence weights
        te: Sum of typicality weights
        ssde: Sum of squared distances weighted by pertinence
    """

    centroid: npt.NDArray[np.float64]
    n: int
    alpha: float
    theta: float
    label: float = -1.0
    true_label: float = -1.0
    created: int = 0
    updated: int = 0
    cf1_pertinence: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    cf1_typicality: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    me: float = 1.0
    te: float = 1.0
    ssde: float = 0.0

    def __post_init__(self):
        """Initialize CF1 vectors if not provided."""
        if len(self.cf1_pertinence) == 0:
            self.cf1_pertinence = self.centroid.copy()
        if len(self.cf1_typicality) == 0:
            self.cf1_typicality = self.centroid.copy()

    def update_centroid(self) -> None:
        """Recalculate centroid from CF1 statistics."""
        denominator = self.alpha * self.te + self.theta * self.me
        if denominator > 0:
            self.centroid = (
                self.alpha * self.cf1_pertinence + self.theta * self.cf1_typicality
            ) / denominator

    def assign_example(
        self, example: Example, pertinence: float, typicality: float, distance: float
    ) -> None:
        """
        Incrementally update micro-cluster with new example.

        Args:
            example: New data point
            pertinence: Membership value
            typicality: Typicality value
            distance: Distance to centroid
        """
        self.n += 1
        self.me += pertinence**self.alpha
        self.te += typicality**self.theta
        self.ssde += pertinence * (distance**2)

        self.cf1_pertinence += example.point * pertinence
        self.cf1_typicality += example.point * typicality

        self.update_centroid()

    def calculate_typicality(
        self, point: npt.NDArray[np.float64], k: int, distance: float
    ) -> float:
        """
        Calculate typicality value for a point.

        Args:
            point: Feature vector
            k: Number of clusters
            distance: Euclidean distance to centroid

        Returns:
            Typicality value in [0, 1]
        """
        gamma = k * (self.ssde / self.me) if self.me > 0 else 1.0

        if gamma == 0:
            return 1.0

        exponent = 1.0 / (self.n - 1) if self.n > 1 else 1.0
        return 1.0 / (1.0 + ((self.theta / gamma) * distance) ** exponent)

    def get_radius(self, multiplier: float = 2.0) -> float:
        """
        Calculate cluster radius.

        Args:
            multiplier: Scaling factor for radius

        Returns:
            Radius value
        """
        if self.n > 0:
            return np.sqrt(self.ssde / self.n) * multiplier
        return 0.0

    @property
    def radius_weighted(self) -> float:
        """Standard radius with multiplier=2.0."""
        return self.get_radius(2.0)

    @property
    def radius_nd(self) -> float:
        """Novelty detection radius (multiplier=1.0)."""
        return self.get_radius(1.0)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.centroid)

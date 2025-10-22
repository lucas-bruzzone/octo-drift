"""
Distance metrics for octo-drift algorithm.
"""

import numpy as np
import numpy.typing as npt
from typing import Union
from .structures import Example, SPFMiC


def euclidean_distance(
    p1: Union[npt.NDArray[np.float64], Example, SPFMiC],
    p2: Union[npt.NDArray[np.float64], Example, SPFMiC],
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point (array, Example, or SPFMiC)
        p2: Second point (array, Example, or SPFMiC)

    Returns:
        Euclidean distance

    Examples:
        >>> p1 = np.array([1.0, 2.0, 3.0])
        >>> p2 = np.array([4.0, 5.0, 6.0])
        >>> euclidean_distance(p1, p2)
        5.196152422706632
    """
    # Extract arrays from objects if needed
    if isinstance(p1, Example):
        p1 = p1.point
    elif isinstance(p1, SPFMiC):
        p1 = p1.centroid

    if isinstance(p2, Example):
        p2 = p2.point
    elif isinstance(p2, SPFMiC):
        p2 = p2.centroid

    # Vectorized computation
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calculate_pertinence(
    point: npt.NDArray[np.float64],
    centroid: npt.NDArray[np.float64],
    fuzzification: float,
) -> float:
    """
    Calculate fuzzy membership (pertinence) value.

    Uses exponential membership function based on distance.

    Args:
        point: Feature vector
        centroid: Cluster center
        fuzzification: Fuzzification parameter (m)

    Returns:
        Membership value in [0, 1]

    Examples:
        >>> point = np.array([1.0, 2.0])
        >>> centroid = np.array([1.0, 2.0])
        >>> calculate_pertinence(point, centroid, 2.0)
        1.0
    """
    distance_sq = np.sum((point - centroid) ** 2)
    return np.exp(-distance_sq / fuzzification)


def batch_euclidean_distance(
    points: npt.NDArray[np.float64], centroid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate Euclidean distances from multiple points to a centroid.

    Args:
        points: Array of shape (n_samples, n_features)
        centroid: Center point of shape (n_features,)

    Returns:
        Array of distances of shape (n_samples,)
    """
    return np.sqrt(np.sum((points - centroid) ** 2, axis=1))


def pairwise_distances(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate pairwise Euclidean distances between all points.

    Args:
        points: Array of shape (n_samples, n_features)

    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n = points.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        distances[i] = batch_euclidean_distance(points, points[i])

    return distances

"""
Fuzzy clustering and related functions for octo-drift.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from .structures import Example, SPFMiC
from .distance import euclidean_distance, batch_euclidean_distance


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering implementation.

    Attributes:
        n_clusters: Number of clusters (K)
        fuzzification: Fuzziness parameter (m)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Seed for reproducibility
        centroids: Cluster centers
        membership_matrix: Fuzzy membership matrix
        labels: Hard cluster assignments
    """

    def __init__(
        self,
        n_clusters: int,
        fuzzification: float = 2.0,
        max_iter: int = 150,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.fuzzification = fuzzification
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids: npt.NDArray[np.float64] = np.array([])
        self.membership_matrix: npt.NDArray[np.float64] = np.array([])
        self.labels: npt.NDArray[np.int32] = np.array([])

    def fit(self, examples: List[Example]) -> "FuzzyCMeans":
        """Fit FCM to examples with improved stability."""
        X = np.array([ex.point for ex in examples])
        n_samples, n_features = X.shape

        # Validação
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) < n_clusters ({self.n_clusters})"
            )

        # Inicialização determinística usando k-means++
        np.random.seed(self.random_state)
        self.centroids = self._init_centroids_kmeanspp(X)

        # Inicializar membership baseado em distâncias aos centroids iniciais
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        distances = np.maximum(distances, 1e-10)

        exponent = 2.0 / (self.fuzzification - 1.0)
        U = 1.0 / np.sum(
            (distances[:, :, np.newaxis] / distances[:, np.newaxis, :]) ** exponent,
            axis=2,
        )

        for iteration in range(self.max_iter):
            U_old = U.copy()

            # Update centroids (vetorizado)
            Um = U**self.fuzzification
            centroid_numerator = Um.T @ X
            centroid_denominator = Um.sum(axis=0, keepdims=True).T
            centroid_denominator = np.maximum(centroid_denominator, 1e-10)
            self.centroids = centroid_numerator / centroid_denominator

            # Update membership (vetorizado com estabilidade)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            distances = np.maximum(distances, 1e-10)

            # Evitar divisão por zero
            exponent = 2.0 / (self.fuzzification - 1.0)
            distance_ratio = distances[:, :, np.newaxis] / distances[:, np.newaxis, :]
            distance_ratio = np.maximum(distance_ratio, 1e-10)

            U = 1.0 / np.sum(distance_ratio**exponent, axis=2)

            # Normalizar para garantir soma = 1
            U = U / U.sum(axis=1, keepdims=True)

            # Convergência
            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.membership_matrix = U
        self.labels = np.argmax(U, axis=1)
        return self

    def _init_centroids_kmeanspp(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Initialize centroids using k-means++ strategy."""
        n_samples = X.shape[0]
        centroids = []

        # Primeiro centroid aleatório
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        # Demais centroids baseados em distâncias
        for _ in range(1, self.n_clusters):
            distances = np.array(
                [
                    min(np.linalg.norm(X[i] - c) for c in centroids)
                    for i in range(n_samples)
                ]
            )
            distances = distances**2
            probabilities = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def get_cluster_members(
        self, cluster_idx: int, examples: List[Example]
    ) -> List[Example]:
        """Get examples assigned to specific cluster."""
        return [ex for i, ex in enumerate(examples) if self.labels[i] == cluster_idx]


def calculate_typicality_matrix(
    membership_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate typicality matrix from membership matrix.

    Typicality = membership / max_membership_per_sample

    Args:
        membership_matrix: Shape (n_samples, n_clusters)

    Returns:
        Typicality matrix of same shape
    """
    max_membership = np.max(membership_matrix, axis=1, keepdims=True)
    max_membership = np.maximum(max_membership, 1e-10)
    return membership_matrix / max_membership


def fuzzy_silhouette(
    clusterer: FuzzyCMeans, examples: List[Example], alpha: float
) -> List[float]:
    """
    Calculate fuzzy silhouette coefficient for each cluster.

    Args:
        clusterer: Fitted FuzzyCMeans object
        examples: List of examples
        alpha: Exponent for membership weighting

    Returns:
        List of silhouette values per cluster
    """
    n_samples = len(examples)
    membership = clusterer.membership_matrix
    labels = clusterer.labels

    silhouettes = []

    for cluster_idx in range(clusterer.n_clusters):
        numerator = 0.0
        denominator = 0.0

        cluster_mask = labels == cluster_idx
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            silhouettes.append(0.0)
            continue

        for j in cluster_indices:
            # Calculate a(j) - average distance within cluster
            a_j = 0.0
            count_within = 0

            for k in range(n_samples):
                if labels[k] == cluster_idx and k != j:
                    dist = euclidean_distance(examples[j], examples[k])
                    a_j += dist
                    count_within += 1

            if count_within > 0:
                a_j /= count_within

            # Calculate b(j) - min average distance to other clusters
            b_j_values = []

            for other_cluster in range(clusterer.n_clusters):
                if other_cluster == cluster_idx:
                    continue

                other_mask = labels == other_cluster
                if not np.any(other_mask):
                    continue

                avg_dist = 0.0
                count = 0
                for k in np.where(other_mask)[0]:
                    dist = euclidean_distance(examples[j], examples[k])
                    avg_dist += dist
                    count += 1

                if count > 0:
                    b_j_values.append(avg_dist / count)

            if not b_j_values:
                continue

            b_j = min(b_j_values)

            # Silhouette coefficient for this example
            s_j = (b_j - a_j) / max(a_j, b_j) if max(a_j, b_j) > 0 else 0.0

            # Get first and second highest membership
            memberships_sorted = np.sort(membership[j])[::-1]
            u_pj = memberships_sorted[0]
            u_qj = memberships_sorted[1] if len(memberships_sorted) > 1 else 0.0

            weight = (u_pj - u_qj) ** alpha
            numerator += weight * s_j
            denominator += weight

        fs = numerator / denominator if denominator > 0 else 0.0
        silhouettes.append(fs)

    return silhouettes


def separate_by_class(examples: List[Example]) -> Dict[float, List[Example]]:
    """
    Separate examples by their true labels.

    Args:
        examples: List of examples

    Returns:
        Dictionary mapping label -> list of examples
    """
    by_class: Dict[float, List[Example]] = {}

    for example in examples:
        label = example.true_label
        if label not in by_class:
            by_class[label] = []
        by_class[label].append(example)

    return by_class


def create_spfmics_from_clusters(
    examples: List[Example],
    clusterer: FuzzyCMeans,
    label: float,
    alpha: float,
    theta: float,
    min_weight: int,
    timestamp: int,
) -> List[SPFMiC]:
    """
    Create SPFMiC micro-clusters from fuzzy clustering results.

    CORREÇÃO: Garante que índices estão alinhados entre examples e clusterer.labels
    """
    spfmics = []
    membership = clusterer.membership_matrix
    typicality = calculate_typicality_matrix(membership)

    # CRÍTICO: Criar mapeamento explícito example -> índice no clustering
    # Isso garante que labels[i] corresponde a examples[i]

    for cluster_idx in range(clusterer.n_clusters):
        # Coletar índices dos exemplos neste cluster
        cluster_member_indices = [
            i for i in range(len(examples)) if clusterer.labels[i] == cluster_idx
        ]

        if len(cluster_member_indices) < min_weight:
            continue

        spfmic = SPFMiC(
            centroid=clusterer.centroids[cluster_idx].copy(),
            n=len(cluster_member_indices),
            alpha=alpha,
            theta=theta,
            label=label,
            created=timestamp,
            updated=timestamp,
        )

        # Aggregate statistics usando índices corretos
        cf1_pert = np.zeros_like(spfmic.centroid)
        cf1_typ = np.zeros_like(spfmic.centroid)
        me = 0.0
        te = 0.0
        ssde = 0.0

        for idx in cluster_member_indices:
            example = examples[idx]
            pert_val = membership[idx, cluster_idx]
            typ_val = typicality[idx, cluster_idx]

            distance = euclidean_distance(example, spfmic.centroid)

            cf1_pert += example.point * pert_val
            cf1_typ += example.point * typ_val
            me += pert_val**alpha
            te += typ_val**theta
            ssde += pert_val * (distance**2)

        spfmic.cf1_pertinence = cf1_pert
        spfmic.cf1_typicality = cf1_typ
        spfmic.me = me
        spfmic.te = te
        spfmic.ssde = ssde

        spfmics.append(spfmic)

    return spfmics

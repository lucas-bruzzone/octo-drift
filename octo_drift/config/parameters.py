"""
Configuration parameters for octo-drift.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class OctoDriftConfig:
    """
    Complete configuration for octo-drift algorithm.

    Clustering parameters:
        k: Clusters per class
        k_short: Clusters for unknown buffer
        fuzzification: Fuzziness parameter (m)
        alpha: Pertinence exponent
        theta: Typicality exponent

    Novelty detection:
        phi: Overlap threshold for novelty
        buffer_threshold: Unknown buffer size trigger (T)
        min_weight_offline: Min cluster size in training
        min_weight_online: Min cluster size in stream

    Incremental learning:
        latency: Label delay
        chunk_size: Batch size for updates
        time_threshold: Removal threshold (ts)
        percent_labeled: Fraction of delayed labels

    Evaluation:
        evaluation_interval: Metrics computation frequency
    """

    # Clustering
    k: int = 4
    k_short: int = 4
    fuzzification: float = 2.0
    alpha: float = 2.0
    theta: float = 1.0

    # Novelty detection
    phi: float = 0.2
    buffer_threshold: int = 40
    min_weight_offline: int = 0
    min_weight_online: int = 15

    # Incremental learning
    latency: int = 10000
    chunk_size: int = 2000
    time_threshold: int = 200
    percent_labeled: float = 1.0

    # Evaluation
    evaluation_interval: int = 1000

    @classmethod
    def from_dict(cls, config_dict: dict) -> "OctoDriftConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "k": self.k,
            "k_short": self.k_short,
            "fuzzification": self.fuzzification,
            "alpha": self.alpha,
            "theta": self.theta,
            "phi": self.phi,
            "buffer_threshold": self.buffer_threshold,
            "min_weight_offline": self.min_weight_offline,
            "min_weight_online": self.min_weight_online,
            "latency": self.latency,
            "chunk_size": self.chunk_size,
            "time_threshold": self.time_threshold,
            "percent_labeled": self.percent_labeled,
            "evaluation_interval": self.evaluation_interval,
        }


@dataclass
class ExperimentConfig:
    """
    Experiment setup for parameter grid search.

    Attributes:
        dataset_name: Dataset identifier
        dataset_path: Path to dataset files
        phi_values: Grid of phi values to test
        buffer_threshold_values: Grid of T values
        k_short_values: Grid of k_short values
        base_config: Base configuration
    """

    dataset_name: str
    dataset_path: str
    phi_values: List[float] = field(default_factory=lambda: [0.2])
    buffer_threshold_values: List[int] = field(default_factory=lambda: [40])
    k_short_values: List[int] = field(default_factory=lambda: [4])
    base_config: OctoDriftConfig = field(default_factory=OctoDriftConfig)

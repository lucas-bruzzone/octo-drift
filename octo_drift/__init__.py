"""
Octo-drift: Novelty detection in data streams with drift adaptation.
"""

__version__ = "0.1.0"

from .config.parameters import OctoDriftConfig, ExperimentConfig
from .pipeline.offline_phase import OfflinePhase
from .pipeline.online_phase import OnlinePhase

__all__ = [
    "OctoDriftConfig",
    "ExperimentConfig",
    "OfflinePhase",
    "OnlinePhase",
]

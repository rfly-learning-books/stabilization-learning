"""Certificate/feature learning components for explicit task adapters.

Task-specific backends share data, version and checkpoint contracts, not an
invented universal loss. Import visual/push explicitly to avoid optional assets.
"""
from .certificate import NeuralCertificate, make_binding

__all__ = ["NeuralCertificate", "make_binding"]

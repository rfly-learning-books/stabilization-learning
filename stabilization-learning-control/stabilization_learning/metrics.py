"""Compatibility facade; implementation is in policies/metric.py."""
from .policies.metric import CholeskyMetric, MetricActor, quadratic_certificate

__all__ = ["CholeskyMetric", "MetricActor", "quadratic_certificate"]

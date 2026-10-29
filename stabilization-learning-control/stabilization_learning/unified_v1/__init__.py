"""Staged training across task-specific scientific backends.

No environments or experiments are started at import time.
"""
from .runner import run_suite, validate_config

__all__ = ["run_suite", "validate_config"]

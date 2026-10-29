"""Separate development diagnostics; archived scientific runs remain immutable."""

# Permit ``python -m extensions.rgb_diagnostic_v5.<stage>`` from the project
# without relying on a machine-specific editable-package installation.
from .common import PROJECT

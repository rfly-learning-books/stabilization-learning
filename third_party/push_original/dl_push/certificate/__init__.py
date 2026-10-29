"""Sampling-first cross-shape CLF control.

This package is the new primary control path described by
``pusht_mjx_requirements``.  It deliberately does not import the historical
prior-policy, observer, DQ, SAC, or MPPI controllers.  The public layers are:

``math``
    Coordinate transforms, the fixed ``C -> Q`` certificate, exact constant
    twist integration, and scalar interval solving.
``library``
    Shape-local contact modes, calibration data, validity masks, and on-disk
    mode libraries.
``transfer``
    Collision-checked release/transit/engage paths between modes.
``solver``
    J/A/P sampled CLF candidate generation and auditable decisions.
``m_metric``
    Teacher-inspired state--shape conditioned SPD metric for the explicitly
    experimental instantaneous-CLF training branch.

The physics adapter, sampling runner, Q search, and set encoder are layered on
top of these pure components so they can be tested without a simulator.
"""

from .math import (
    STAGES,
    certificate,
    exact_se2_step,
    minimum_linear_input,
    q_from_parameters,
    response_in_error_frame,
    stage_projection,
)
from .library import ModeLibrary, boundary_mode_library, fit_mode_response, support_mode_library, fixed_mode_library
from .solver import CLFConfig, CLFSolver, Decision, ModeCandidate
from .q_generator import CertificateGenerator, QConfig, SearchConfig, search_q, SetQGenerator
from .m_metric import MetricConfig, StateShapeMetric, spd_from_parameters
from .selection import SelectionReport, attach_transfer_graph, direction_coverage, select_modes

__all__ = [
    "STAGES",
    "certificate",
    "exact_se2_step",
    "minimum_linear_input",
    "q_from_parameters",
    "response_in_error_frame",
    "stage_projection",
    "ModeLibrary",
    "boundary_mode_library",
    "support_mode_library",
    "fixed_mode_library",
    "fit_mode_response",
    "CLFConfig",
    "CLFSolver",
    "Decision",
    "ModeCandidate",
    "CertificateGenerator",
    "QConfig",
    "SearchConfig",
    "search_q",
    "SetQGenerator",
    "MetricConfig",
    "StateShapeMetric",
    "spd_from_parameters",
    "SelectionReport",
    "attach_transfer_graph",
    "direction_coverage",
    "select_modes",
]

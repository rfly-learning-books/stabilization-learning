"""Motion-authority and terminal-capability mode selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import ModeLibrary


@dataclass
class SelectionReport:
    selected_indices: np.ndarray
    direction_coverage_before: float
    direction_coverage_after: float
    positive_rotation_modes: int
    negative_rotation_modes: int
    near_translation_modes: int
    terminal_position_capable: int

    def to_dict(self):
        return {
            "selected_indices": self.selected_indices.tolist(),
            "direction_coverage_before": float(self.direction_coverage_before),
            "direction_coverage_after": float(self.direction_coverage_after),
            "positive_rotation_modes": int(self.positive_rotation_modes),
            "negative_rotation_modes": int(self.negative_rotation_modes),
            "near_translation_modes": int(self.near_translation_modes),
            "terminal_position_capable": int(self.terminal_position_capable),
        }


def _sphere_directions(count=4096):
    # Deterministic Fibonacci sphere, avoiding a dependency on a sampler.
    index = np.arange(int(count), dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / int(count)
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack((radius * np.cos(golden * index), radius * np.sin(golden * index), z))


def direction_coverage(library: ModeLibrary, indices=None, feature_length=.10, samples=2048):
    """Approximate the response-direction support margin of a mode subset."""

    if indices is None:
        indices = library.active_indices
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return 0.0
    responses = library.c[:, indices].T * np.array([1.0, 1.0, feature_length])
    responses *= library.u_max[indices, None]
    norms = np.linalg.norm(responses, axis=1)
    responses = responses[norms > 1e-10]
    if len(responses) == 0:
        return 0.0
    responses = responses / np.maximum(np.linalg.norm(responses, axis=1, keepdims=True), 1e-12)
    directions = _sphere_directions(samples)
    support = np.maximum(0.0, directions @ responses.T).max(axis=1)
    return float(support.min())


def select_modes(library: ModeLibrary, target_count=8, feature_length=.10, samples=2048):
    """Greedily retain motion coverage and both rotation signs.

    The selection is performed only from calibrated valid modes.  It never
    synthesizes a response by adding two contact modes.
    """

    active = library.active_indices
    if len(active) == 0:
        raise ValueError("cannot select modes before calibration")
    target = min(int(target_count), len(active))
    responses = library.c[:, active].T * np.array([1.0, 1.0, feature_length])
    responses *= library.u_max[active, None]
    selected = []

    def add_first(predicate):
        candidates = [int(index) for index in active if predicate(index) and int(index) not in selected]
        if candidates:
            selected.append(max(candidates, key=lambda index: (np.linalg.norm(library.c[:, index]), -index)))

    # Preserve angular authority and near-pure translation where possible.
    add_first(lambda index: library.c[2, index] > 1e-5)
    add_first(lambda index: library.c[2, index] < -1e-5)
    add_first(lambda index: abs(library.c[2, index]) <= .1 * max(np.linalg.norm(library.c[:2, index]), 1e-8))
    while len(selected) < target:
        best = None
        best_score = -np.inf
        for index in active:
            index = int(index)
            if index in selected:
                continue
            trial = selected + [index]
            score = direction_coverage(library, trial, feature_length, samples=max(256, samples // 4))
            tie = float(np.linalg.norm(responses[np.flatnonzero(active == index)[0]]))
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tie > (best[1] if best else -np.inf)):
                best = (index, tie)
                best_score = score
        if best is None:
            break
        selected.append(best[0])
    selected = np.asarray(selected[:target], dtype=np.int64)
    report = SelectionReport(
        selected_indices=selected,
        direction_coverage_before=direction_coverage(library, active, feature_length, samples),
        direction_coverage_after=direction_coverage(library, selected, feature_length, samples),
        positive_rotation_modes=int(np.sum(library.c[2, selected] > 1e-5)),
        negative_rotation_modes=int(np.sum(library.c[2, selected] < -1e-5)),
        near_translation_modes=int(np.sum(np.abs(library.c[2, selected]) <= .1 * np.maximum(np.linalg.norm(library.c[:2, selected], axis=0), 1e-8))),
        terminal_position_capable=int(np.sum(np.linalg.norm(library.c[:2, selected], axis=0) > 1e-5)),
    )
    return library.subset(selected), report


def attach_transfer_graph(library: ModeLibrary, graph):
    """Return a copy with a previously verified transfer graph attached."""

    if graph.distance.shape != (library.count, library.count):
        raise ValueError("transfer graph mode count differs from library")
    result = library.copy()
    result.transition_distance = graph.distance.copy()
    result.transition_time = graph.time.copy()
    result.transition_valid = graph.valid.copy()
    result.paths = graph.paths
    result.metadata = dict(result.metadata)
    result.metadata["transfer_metadata"] = dict(graph.metadata)
    return result


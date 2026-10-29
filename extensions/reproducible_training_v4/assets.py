"""Freeze assets that are otherwise first opened by the historical RGB test."""
from pathlib import Path

from stabilization_learning.unified_v1.runner import sha256


# The retained evaluator checks this same scientific identity. Keep a literal
# here so CPU preflight does not import or initialize the CUDA RGB evaluator.
HISTORICAL_BEST_SHA256 = "3edf4756000daa3373da2e2988c1ad5090201fdd45215aed2092bf1edaad6bca"
HISTORICAL_VARIANTS = frozenset({"historical_trained_reference", "historical_trained_identity"})


def _absolute_file(value, label):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(label + " must be an absolute existing file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(label + " must be an absolute existing file")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def historical_sources(project, config):
    """Validate requested references and return all declared asset fingerprints.

    Historical comparison is optional and may use one or multiple RGB arms.
    When requested, its actual fixed best.pt is always frozen. An optional
    source_paths declaration must include that file and every separately named
    original checkpoint; every declared source is resolved and hashed.
    """
    requested = False
    for job in config.get("jobs", []):
        enabled = job.get("config", {}).get("include_historical_reference", False)
        if type(enabled) is not bool:
            raise ValueError("include_historical_reference must be boolean")
        variants = job.get("test_variants", ["selected", "initial", "identity"])
        historical = bool(HISTORICAL_VARIANTS.intersection(variants))
        if (enabled or historical) and job.get("backend") != "rgb_legacy":
            raise ValueError("Historical RGB references require the rgb_legacy backend")
        if enabled != historical:
            raise ValueError("Historical RGB variants and include_historical_reference must agree")
        requested |= historical
    if not requested:
        return {}

    actual = (Path(project).resolve() / "assets/original_visual_initialization/best.pt").resolve()
    if not actual.is_file():
        raise FileNotFoundError(actual)
    digest = sha256(actual)
    if digest != HISTORICAL_BEST_SHA256:
        raise ValueError("Historical best checkpoint differs from the retained evaluator hash")
    result = {str(actual): digest}
    spec = config.get("historical_reference", {})
    if not isinstance(spec, dict):
        raise ValueError("historical_reference must be a mapping")
    if "checkpoint_sha256" in spec and spec["checkpoint_sha256"] != HISTORICAL_BEST_SHA256:
        raise ValueError("Declared historical checkpoint hash differs from the retained evaluator hash")
    if "checkpoint" in spec:
        declared_checkpoint = _absolute_file(spec["checkpoint"], "historical checkpoint")
        if declared_checkpoint != actual:
            raise ValueError("Historical checkpoint must be the fixed asset actually loaded by the evaluator")

    original = None
    if "original_checkpoint" in spec:
        original = _absolute_file(spec["original_checkpoint"], "original historical checkpoint")
        original_hash = sha256(original)
        if original_hash != HISTORICAL_BEST_SHA256:
            raise ValueError("Original historical checkpoint differs from the retained evaluator hash")
        result[str(original)] = original_hash
    if "source_paths" in spec:
        values = spec["source_paths"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("historical source_paths must be a nonempty absolute file list")
        declared = [_absolute_file(value, "historical source") for value in values]
        if len(set(declared)) != len(declared):
            raise ValueError("Historical source_paths contains duplicate resolved files")
        required = {actual} | ({original} if original is not None else set())
        if not required.issubset(declared):
            raise ValueError("Historical source_paths must include the actual and declared original checkpoints")
        for path in declared:
            result[str(path)] = sha256(path)
    return result

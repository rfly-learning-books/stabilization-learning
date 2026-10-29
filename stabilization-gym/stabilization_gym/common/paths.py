"""Explicit external asset configuration; independent of package depth."""
import json
import os
from pathlib import Path


def asset_path(name):
    override = os.environ.get("STABILIZATION_ASSET_CONFIG")
    if override:
        config = Path(override).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"STABILIZATION_ASSET_CONFIG does not exist: {config}")
    else:
        config = next((p / "assets.json" for p in Path(__file__).resolve().parents
                       if (p / "assets.json").is_file()), None)
    if config is None:
        # A standalone installation still supports all asset-free tasks.
        # Optional assets are validated only by the task that actually uses them.
        root = Path(os.environ.get("STABILIZATION_ASSET_ROOT", Path.home() / ".stabilization" / "assets"))
        defaults = {"legacy_workspace": root, "push_source": root / "push_source",
                    "push_library": root / "push_library", "visual_legacy_project": root / "visual_legacy_project",
                    "fastdepth": root / "fastdepth"}
        if name not in defaults:
            raise KeyError(f"Unknown asset key {name!r}")
        return defaults[name]
    data = json.loads(config.read_text(encoding="utf-8-sig"))
    if name not in data:
        raise KeyError(f"Asset key {name!r} missing from {config}")
    result = Path(data[name]).expanduser()
    return result if result.is_absolute() else (config.parent / result).resolve()

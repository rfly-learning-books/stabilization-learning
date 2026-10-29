"""Resolve optional SO-101 visuals without searching other projects or SDKs."""
import os
import html
from pathlib import Path

def texture_path(filename):
    """Resolve packaged textures relative to this module, including XML escaping."""
    path=Path(__file__).parent/filename
    if not path.is_file():raise FileNotFoundError(f"Packaged texture missing: {filename}")
    return html.escape(str(path.resolve()),quote=True)

def resolve_so101_mesh_dir(required_names):
    mode = os.environ.get("SO101_VISUAL_MODE", "primitive").strip().lower()
    if mode not in {"primitive", "auto", "mesh"}:
        raise ValueError("SO101_VISUAL_MODE must be primitive, auto, or mesh")
    if mode == "primitive":
        return None
    override = os.environ.get("SO101_MESH_DIR")
    candidates = ([Path(override)] if override else []) + [Path(__file__).parent / "assets" / "so101"]
    for path in candidates:
        if all((path / name).is_file() for name in required_names):
            return str(path.resolve())
    if mode == "mesh":
        raise FileNotFoundError("Optional SO-101 meshes missing; set SO101_MESH_DIR or use SO101_VISUAL_MODE=primitive")
    return None

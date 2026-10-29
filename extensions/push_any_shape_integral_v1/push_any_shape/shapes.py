"""The preserved 32 outlines plus one genuinely rotation-invariant Disk.

Loading this catalog performs file/geometry work only.  It never constructs a
simulator.  The disk's polygon is for display/overlap; physical contact uses an
analytic MuJoCo cylinder, and its task ignores yaw.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from shapely.geometry import Point, shape


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRARY_ROOT = PROJECT_ROOT / "assets" / "push_geometric32"
DISK_RADIUS_M = .015
DISK_NAME = "Disk"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    polygon: object
    yaw_free: bool
    analytic_radius_m: float | None
    source: dict
    contact_backend_supported: bool = False

    @property
    def task_mask(self):
        return (1., 1., 0. if self.yaw_free else 1.)

    @property
    def poly(self):
        return self.polygon

    @property
    def rotation_free(self):
        return self.yaw_free


def catalog(library_root=DEFAULT_LIBRARY_ROOT):
    """Return exactly 32 preserved names and Disk; no near-circle aliasing."""
    root = Path(library_root).resolve()
    names = sorted(p.parent.name for p in root.glob("*/report.json"))
    if len(names) != 32 or len(set(names)) != 32 or DISK_NAME in names:
        raise ValueError("Expected the intact 32-shape geometric source catalog")
    result = {}
    for name in names:
        polygon_path = root / name / "polygon.json"
        polygon = shape(json.loads(polygon_path.read_text(encoding="utf-8-sig")))
        if polygon.geom_type != "Polygon" or not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Invalid preserved outline: {name}")
        result[name] = ShapeSpec(name, polygon, False, None, {
            "kind": "preserved_original_polygon",
            "path": str(polygon_path), "sha256": file_sha256(polygon_path),
            "contact_support": "No physical backend for these 32 shapes in this extension",
        })
    result[DISK_NAME] = ShapeSpec(DISK_NAME, Point(0., 0.).buffer(DISK_RADIUS_M, resolution=128),
        True, DISK_RADIUS_M, {
            "kind": "analytic_circle",
            "radius_m": DISK_RADIUS_M,
            "display_polygon_segments": 512,
            "yaw_semantics": "continuous rotation symmetry; yaw is observed but is not a target error",
            "physical_geometry": "MuJoCo cylinder, exact radius parameter",
        }, True)
    return result


def load_shape(name, library_root=DEFAULT_LIBRARY_ROOT):
    try:
        return catalog(library_root)[name]
    except KeyError as exc:
        raise ValueError(f"Unknown PushAnyShape catalog entry: {name!r}") from exc


load_catalog = catalog


class ShapeCatalog:
    """Small explicit catalog interface shared by numeric/contact tasks."""
    def __init__(self, project_root=None, *, library_root=None):
        if library_root is None:
            library_root = DEFAULT_LIBRARY_ROOT if project_root is None else Path(project_root) / "assets" / "push_geometric32"
        self.library_root = Path(library_root).resolve()
        self.specs = catalog(self.library_root)
        self.names = list(self.specs)

    def get(self, name):
        if name not in self.specs:
            raise ValueError(f"Unknown PushAnyShape entry: {name!r}")
        return self.specs[name]

    def __len__(self):
        return len(self.names)

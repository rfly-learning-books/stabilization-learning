"""Cross-shape contact-mode libraries and boundary sampling.

The mode library is the bridge between real sampled contact experiments and
the low-dimensional CLF controller.  Every response stays in raw units; no
column is normalized to unit length because translation and rotation authority
must remain visible to the certificate and terminal-mode audits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


def _array(value, shape=None, name="array"):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass
class ModeLibrary:
    """Verified shape-local modes and their optional transfer graph."""

    shape: str
    c: np.ndarray
    contact_points: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    valid: np.ndarray
    face_normals: np.ndarray | None = None
    face_mask: np.ndarray | None = None
    contact_offset: np.ndarray | None = None
    mode_type: np.ndarray | None = None
    fit_error: np.ndarray = field(default_factory=lambda: np.empty(0))
    hold_rate: np.ndarray = field(default_factory=lambda: np.empty(0))
    metadata: dict = field(default_factory=dict)
    transition_distance: np.ndarray | None = None
    transition_time: np.ndarray | None = None
    transition_valid: np.ndarray | None = None
    paths: list[list[list[float]]] | None = None

    schema: int = 1

    def __post_init__(self):
        self.shape = str(self.shape)
        self.c = _array(self.c, name="c")
        if self.c.ndim != 2 or self.c.shape[0] != 3 or self.c.shape[1] < 1:
            raise ValueError("c must have shape (3,N)")
        count = self.c.shape[1]
        self.contact_points = _array(self.contact_points, (count, 2), "contact_points")
        self.normals = _array(self.normals, (count, 2), "normals")
        self.tangents = _array(self.tangents, (count, 2), "tangents")
        self.u_min = _array(self.u_min, (count,), "u_min")
        self.u_max = _array(self.u_max, (count,), "u_max")
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.valid.shape != (count,):
            raise ValueError("valid must have shape (N,)")
        if (self.u_min < 0).any() or (self.u_max < self.u_min).any():
            raise ValueError("invalid mode speed domains")
        if self.face_normals is None:
            self.face_normals = np.zeros((count, 2, 2), dtype=np.float64)
            self.face_normals[:, 0] = self.normals
        else:
            self.face_normals = _array(self.face_normals, (count, 2, 2), "face_normals")
        if self.face_mask is None:
            self.face_mask = np.zeros((count, 2), dtype=bool)
            self.face_mask[:, 0] = True
        else:
            self.face_mask = np.asarray(self.face_mask, dtype=bool)
            if self.face_mask.shape != (count, 2):
                raise ValueError("face_mask must have shape (N,2)")
        active_face_normals = self.face_normals[self.face_mask]
        if len(active_face_normals) and not np.allclose(
            np.linalg.norm(active_face_normals, axis=1), 1.0, atol=1e-5, rtol=0
        ):
            raise ValueError("active face normals must be unit length")
        if self.contact_offset is None:
            self.contact_offset = np.full(
                count,
                float(self.metadata.get("contact_center_offset", .020)),
                dtype=np.float64,
            )
        else:
            self.contact_offset = _array(self.contact_offset, (count,), "contact_offset")
        if (self.contact_offset <= 0).any():
            raise ValueError("contact_offset must be positive")
        if self.mode_type is None:
            self.mode_type = np.where(self.face_mask[:, 1], "corner_pair", "single")
        else:
            self.mode_type = np.asarray(self.mode_type, dtype="U32")
            if self.mode_type.shape != (count,):
                raise ValueError("mode_type must have shape (N,)")
        for name in ("normals", "tangents"):
            values = getattr(self, name)
            lengths = np.linalg.norm(values, axis=1)
            if not np.allclose(lengths[self.valid], 1.0, atol=1e-5, rtol=0):
                raise ValueError(f"valid {name} rows must be unit length")
        if np.any(np.abs((self.normals * self.tangents).sum(axis=1)[self.valid]) > 1e-4):
            raise ValueError("valid mode normals and tangents must be orthogonal")
        if self.fit_error.size == 0:
            self.fit_error = np.full(count, np.inf, dtype=np.float64)
        else:
            self.fit_error = np.asarray(self.fit_error, dtype=np.float64)
            if self.fit_error.shape != (count,) or np.isnan(self.fit_error).any() or (self.fit_error < 0).any():
                raise ValueError("fit_error must have shape (N) and be nonnegative")
        if self.hold_rate.size == 0:
            self.hold_rate = np.zeros(count, dtype=np.float64)
        else:
            self.hold_rate = _array(self.hold_rate, (count,), "hold_rate")
        if np.any((self.hold_rate < 0) | (self.hold_rate > 1)):
            raise ValueError("hold_rate must be in [0,1]")
        if self.transition_distance is not None:
            self.transition_distance = _array(
                self.transition_distance, (count, count), "transition_distance"
            )
        if self.transition_time is not None:
            self.transition_time = _array(self.transition_time, (count, count), "transition_time")
        if self.transition_valid is not None:
            self.transition_valid = np.asarray(self.transition_valid, dtype=bool)
            if self.transition_valid.shape != (count, count):
                raise ValueError("transition_valid must have shape (N,N)")
        if self.paths is not None and len(self.paths) != count:
            raise ValueError("paths must have one row per source mode")
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("shape", self.shape)
        self.metadata.setdefault("mode_count", count)

    @property
    def count(self):
        return int(self.c.shape[1])

    @property
    def active_indices(self):
        return np.flatnonzero(self.valid)

    @property
    def fingerprint(self):
        payload = {
            "shape": self.shape,
            "c": self.c.tolist(),
            "contact_points": self.contact_points.tolist(),
            "normals": self.normals.tolist(),
            "u_min": self.u_min.tolist(),
            "u_max": self.u_max.tolist(),
            "valid": self.valid.astype(int).tolist(),
            "face_normals": self.face_normals.tolist(),
            "face_mask": self.face_mask.astype(int).tolist(),
            "contact_offset": self.contact_offset.tolist(),
            "mode_type": self.mode_type.tolist(),
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def copy(self):
        return ModeLibrary(
            shape=self.shape,
            c=self.c.copy(),
            contact_points=self.contact_points.copy(),
            normals=self.normals.copy(),
            tangents=self.tangents.copy(),
            u_min=self.u_min.copy(),
            u_max=self.u_max.copy(),
            valid=self.valid.copy(),
            face_normals=self.face_normals.copy(),
            face_mask=self.face_mask.copy(),
            contact_offset=self.contact_offset.copy(),
            mode_type=self.mode_type.copy(),
            fit_error=self.fit_error.copy(),
            hold_rate=self.hold_rate.copy(),
            metadata=dict(self.metadata),
            transition_distance=None if self.transition_distance is None else self.transition_distance.copy(),
            transition_time=None if self.transition_time is None else self.transition_time.copy(),
            transition_valid=None if self.transition_valid is None else self.transition_valid.copy(),
            paths=None if self.paths is None else json.loads(json.dumps(self.paths)),
        )

    def subset(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= self.count):
            raise ValueError("invalid mode subset")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("mode subset contains duplicates")
        transition_distance = (
            None if self.transition_distance is None
            else self.transition_distance[np.ix_(indices, indices)]
        )
        transition_time = (
            None if self.transition_time is None
            else self.transition_time[np.ix_(indices, indices)]
        )
        transition_valid = (
            None if self.transition_valid is None
            else self.transition_valid[np.ix_(indices, indices)]
        )
        paths = None
        if self.paths is not None:
            paths = [[self.paths[int(i)][int(j)] for j in indices] for i in indices]
        metadata = dict(self.metadata)
        metadata["parent_fingerprint"] = self.fingerprint
        metadata["selected_indices"] = indices.tolist()
        return ModeLibrary(
            shape=self.shape,
            c=self.c[:, indices],
            contact_points=self.contact_points[indices],
            normals=self.normals[indices],
            tangents=self.tangents[indices],
            u_min=self.u_min[indices],
            u_max=self.u_max[indices],
            valid=self.valid[indices],
            face_normals=self.face_normals[indices],
            face_mask=self.face_mask[indices],
            contact_offset=self.contact_offset[indices],
            mode_type=self.mode_type[indices],
            fit_error=self.fit_error[indices],
            hold_rate=self.hold_rate[indices],
            metadata=metadata,
            transition_distance=transition_distance,
            transition_time=transition_time,
            transition_valid=transition_valid,
            paths=paths,
        )

    def save(self, directory, overwrite=False):
        """Write JSON metadata and numeric arrays with a content fingerprint."""

        directory = Path(directory).resolve()
        if directory.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite mode library: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            c=self.c,
            contact_points=self.contact_points,
            normals=self.normals,
            tangents=self.tangents,
            u_min=self.u_min,
            u_max=self.u_max,
            valid=self.valid,
            face_normals=self.face_normals,
            face_mask=self.face_mask,
            contact_offset=self.contact_offset,
            mode_type=self.mode_type,
            fit_error=self.fit_error,
            hold_rate=self.hold_rate,
        )
        if self.transition_distance is not None:
            arrays["transition_distance"] = self.transition_distance
        if self.transition_time is not None:
            arrays["transition_time"] = self.transition_time
        if self.transition_valid is not None:
            arrays["transition_valid"] = self.transition_valid
        np.savez_compressed(directory / "library.npz", **arrays)
        record = dict(
            schema=self.schema,
            shape=self.shape,
            metadata=self.metadata,
            fingerprint=self.fingerprint,
            paths=self.paths,
        )
        (directory / "library.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n"
        )
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        record = json.loads((directory / "library.json").read_text())
        with np.load(directory / "library.npz", allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        library = cls(
            shape=record["shape"],
            c=arrays.pop("c"),
            contact_points=arrays.pop("contact_points"),
            normals=arrays.pop("normals"),
            tangents=arrays.pop("tangents"),
            u_min=arrays.pop("u_min"),
            u_max=arrays.pop("u_max"),
            valid=arrays.pop("valid"),
            face_normals=arrays.pop("face_normals", None),
            face_mask=arrays.pop("face_mask", None),
            contact_offset=arrays.pop("contact_offset", None),
            mode_type=arrays.pop("mode_type", None),
            fit_error=arrays.pop("fit_error"),
            hold_rate=arrays.pop("hold_rate"),
            metadata=record.get("metadata", {}),
            transition_distance=arrays.pop("transition_distance", None),
            transition_time=arrays.pop("transition_time", None),
            transition_valid=arrays.pop("transition_valid", None),
            paths=record.get("paths"),
        )
        if arrays or library.fingerprint != record.get("fingerprint"):
            raise ValueError("mode library fingerprint or array contract mismatch")
        return library


def _boundary_rings(polygon):
    if polygon.geom_type == "Polygon":
        polygons = [polygon]
    elif polygon.geom_type == "MultiPolygon":
        polygons = list(polygon.geoms)
    else:
        raise ValueError(f"expected Polygon or MultiPolygon, got {polygon.geom_type}")
    for item in polygons:
        yield item.exterior
        yield from item.interiors


def _ring_normal(ring, distance, polygon, epsilon):
    from shapely.geometry import Point

    length = float(ring.length)
    delta = min(max(length * 1e-5, epsilon), length / 4.0)
    before = np.asarray(ring.interpolate((distance - delta) % length).coords[0], dtype=float)
    after = np.asarray(ring.interpolate((distance + delta) % length).coords[0], dtype=float)
    tangent = after - before
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    first = np.array([-tangent[1], tangent[0]], dtype=float)
    point = np.asarray(ring.interpolate(distance).coords[0], dtype=float)
    probe = point + first * epsilon
    # A point outside the material is the outward normal, including a hole.
    if polygon.contains(Point(probe)):
        normal = -first
    else:
        normal = first
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    return point, normal, tangent


def boundary_mode_library(
    polygon,
    shape,
    count=64,
    pusher_radius=0.018,
    clearance=0.002,
    u_max=0.06,
    u_min=0.0,
    metadata=None,
):
    """Sample an arc-length balanced set of external contact modes.

    Boundary sampling covers every polygon component and hole.  It does not
    use center rays, so non-star-shaped C/L/Z geometries retain hidden sides.
    The returned responses are zero until real calibration fills them.
    """

    from shapely.geometry import Point

    if int(count) < 1:
        raise ValueError("count must be positive")
    if min(float(pusher_radius), float(clearance)) < 0 or float(u_max) <= 0:
        raise ValueError("invalid pusher or speed settings")
    rings = list(_boundary_rings(polygon))
    lengths = np.asarray([float(ring.length) for ring in rings], dtype=float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("polygon has an empty boundary ring")
    # Guarantee at least one sample per ring, then distribute remaining points
    # by arc length using a largest-remainder allocation.
    allocation = np.ones(len(rings), dtype=int)
    remaining = int(count) - len(rings)
    if remaining > 0:
        raw = remaining * lengths / lengths.sum()
        allocation += np.floor(raw).astype(int)
        for index in np.argsort(-(raw - np.floor(raw)))[: remaining - int(np.floor(raw).sum())]:
            allocation[index] += 1
    points, normals, tangents = [], [], []
    for ring, amount in zip(rings, allocation):
        for distance in (np.arange(amount, dtype=float) + 0.5) * float(ring.length) / amount:
            point, normal, tangent = _ring_normal(ring, distance, polygon, max(1e-6, clearance / 4.0))
            # Contact point must be on the material boundary; Point is used to
            # catch malformed geometry that produces a non-finite interpolation.
            if not np.isfinite(point).all() or not Point(point).is_valid:
                raise ValueError("non-finite boundary candidate")
            points.append(point)
            normals.append(normal)
            tangents.append(tangent)
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    tangents = np.asarray(tangents, dtype=float)
    candidate_metadata = dict(metadata or {})
    candidate_metadata.update(
        dict(
            sampling="boundary_arc_length",
            requested_count=int(count),
            actual_count=len(points),
            pusher_radius=float(pusher_radius),
            clearance=float(clearance),
            contact_center_offset=float(pusher_radius + clearance),
        )
    )
    return ModeLibrary(
        shape=shape,
        c=np.zeros((3, len(points)), dtype=float),
        contact_points=points,
        normals=normals,
        tangents=tangents,
        u_min=np.full(len(points), float(u_min), dtype=float),
        u_max=np.full(len(points), float(u_max), dtype=float),
        valid=np.ones(len(points), dtype=bool),
        metadata=candidate_metadata,
    )


def support_mode_library(
    polygon,
    shape,
    directions=8,
    pusher_radius=0.018,
    clearance=0.002,
    u_max=0.06,
    u_min=0.0,
    metadata=None,
):
    """Create eight directly specified modes on exposed convex-hull faces.

    The modes are fixed before calibration: sample the true polygon's exposed
    boundary segments (not the concave indentation edges), distribute eight
    points by segment length, and use each segment's measured outward normal.
    This gives Plus/H/T useful face contacts while retaining the true polygon;
    no convex hull is written into the collision model.  The full arc-length
    sampler remains available for exploratory mode discovery.
    """

    if int(directions) != 8:
        raise ValueError("the fixed certificate mode contract requires eight directions")
    from shapely.geometry import LineString, Point

    hull = polygon.convex_hull
    segments = []
    for ring in _boundary_rings(polygon):
        if ring.is_ring is False:
            continue
        coordinates = list(ring.coords)
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            segment = LineString([start, end])
            if segment.length <= 1e-10 or not hull.boundary.covers(segment):
                continue
            midpoint = np.asarray(segment.interpolate(.5, normalized=True).coords[0], dtype=float)
            tangent = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
            tangent /= max(np.linalg.norm(tangent), 1e-12)
            normal_a = np.array([-tangent[1], tangent[0]], dtype=float)
            probe = Point(midpoint + normal_a * max(clearance / 4.0, 1e-6))
            normal = -normal_a if polygon.contains(probe) else normal_a
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            segments.append((segment.length, np.asarray(start, dtype=float), np.asarray(end, dtype=float), normal, tangent))
    if not segments:
        raise ValueError("polygon has no exposed boundary faces for fixed modes")
    lengths = np.asarray([segment[0] for segment in segments], dtype=float)
    allocation = np.ones(len(segments), dtype=int)
    remaining = 8 - len(segments)
    if remaining < 0:
        # Keep the longest exposed faces; the fixed count is an explicit
        # contract and short sub-faces would otherwise crowd out all sides.
        order = np.argsort(-lengths)[:8]
        segments = [segments[int(index)] for index in order]
        lengths = lengths[order]
        allocation = np.ones(8, dtype=int)
        remaining = 0
    if remaining > 0:
        raw = remaining * lengths / lengths.sum()
        allocation += np.floor(raw).astype(int)
        remainder = remaining - int(np.floor(raw).sum())
        for index in np.argsort(-(raw - np.floor(raw)))[:remainder]:
            allocation[int(index)] += 1
    points, normals, tangents = [], [], []
    for (length, start, end, normal, tangent), amount in zip(segments, allocation):
        for fraction in (np.arange(amount, dtype=float) + .5) / amount:
            points.append(start + fraction * (end - start))
            normals.append(normal)
            tangents.append(tangent)
    candidate_metadata = dict(metadata or {})
    candidate_metadata.update(
        dict(
            sampling="fixed_exterior_support_directions",
            mode_count=8,
            exposed_segments=len(segments),
            pusher_radius=float(pusher_radius),
            clearance=float(clearance),
            contact_center_offset=float(pusher_radius + clearance),
        )
    )
    return ModeLibrary(
        shape=shape,
        c=np.zeros((3, 8), dtype=float),
        contact_points=np.asarray(points, dtype=float),
        normals=np.asarray(normals, dtype=float),
        tangents=np.asarray(tangents, dtype=float),
        u_min=np.full(8, float(u_min), dtype=float),
        u_max=np.full(8, float(u_max), dtype=float),
        valid=np.ones(8, dtype=bool),
        metadata=candidate_metadata,
    )


def fixed_mode_library(
    polygon,
    shape,
    mode_count=8,
    pusher_radius=0.018,
    clearance=0.002,
    u_max=0.06,
    u_min=0.0,
    max_corner_modes=4,
    metadata=None,
):
    """Define eight fixed faces/corners before any physical calibration.

    Convex-hull-exposed face samples provide ordinary single-face modes.  True
    concave vertices are added as explicit two-face corner candidates when the
    two outward face normals admit a finite exterior bisector.  Their
    ``contact_offset`` is enlarged by the bisector angle so a spherical pusher
    can touch both faces simultaneously.  Whether the pair is physically
    stable is decided later by :class:`ModeCalibrator`; a small measured
    angular response is retained as part of ``c_i`` rather than rejected.
    """

    from shapely.geometry import LineString, Point

    if int(mode_count) != 8:
        raise ValueError("fixed mode library requires exactly eight candidates")
    if min(float(pusher_radius), float(clearance)) <= 0 or float(u_max) <= 0:
        raise ValueError("invalid fixed mode geometry or speed")

    def segment_normal(start, end):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        tangent = end - start
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        first = np.array([-tangent[1], tangent[0]], dtype=float)
        midpoint = .5 * (start + end)
        probe = Point(midpoint + first * max(clearance / 4.0, 1e-6))
        normal = -first if polygon.contains(probe) else first
        return normal / max(np.linalg.norm(normal), 1e-12)

    # First collect concave vertices from the material's exterior rings.  A
    # vertex is concave when its signed turn points into the free-space notch.
    corners = []
    polygons = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
    for item in polygons:
        ring = item.exterior
        coordinates = [np.asarray(value, dtype=float) for value in ring.coords[:-1]]
        ccw = bool(ring.is_ccw)
        for index, vertex in enumerate(coordinates):
            previous = coordinates[index - 1]
            following = coordinates[(index + 1) % len(coordinates)]
            incoming = vertex - previous
            outgoing = following - vertex
            turn = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
            concave = turn < -1e-10 if ccw else turn > 1e-10
            if not concave:
                continue
            normal_previous = segment_normal(previous, vertex)
            normal_following = segment_normal(vertex, following)
            bisector_sum = normal_previous + normal_following
            norm = float(np.linalg.norm(bisector_sum))
            if norm <= 1e-8:
                continue
            bisector = bisector_sum / norm
            cosine = float(np.dot(bisector, normal_previous))
            if cosine <= 1e-3:
                continue
            corners.append({
                "point": vertex,
                "normal": bisector,
                "tangent": np.array([-bisector[1], bisector[0]], dtype=float),
                "faces": np.stack([normal_previous, normal_following]),
                "offset": (float(pusher_radius) + float(clearance)) / cosine,
                "type": "corner_pair",
            })
    if len(corners) > int(max_corner_modes):
        indices = np.linspace(0, len(corners) - 1, int(max_corner_modes), dtype=int)
        corners = [corners[int(index)] for index in indices]

    # Ordinary candidates come from actual exterior segments which lie on the
    # convex hull boundary.  Concave boundary segments are intentionally not
    # silently treated as single-face support modes.
    hull = polygon.convex_hull
    segments = []
    for item in polygons:
        coordinates = list(item.exterior.coords)
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            line = LineString([start, end])
            if line.length <= 1e-10 or not hull.boundary.covers(line):
                continue
            normal = segment_normal(start, end)
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            segments.append({
                "length": float(line.length),
                "start": np.asarray(start, dtype=float),
                "end": np.asarray(end, dtype=float),
                "normal": normal,
                "tangent": tangent,
                "type": "single",
            })
    if not segments:
        raise ValueError("polygon has no usable exterior face candidates")
    remaining = 8 - len(corners)
    if remaining <= 0:
        face_candidates = []
    elif len(segments) <= remaining:
        face_candidates = segments
    else:
        # Keep one longest face per approximate normal direction before filling
        # remaining slots by length; this is deterministic and preserves all
        # cardinal directions on rectilinear shapes.
        chosen = []
        for sign in ((1., 0.), (0., 1.), (-1., 0.), (0., -1.)):
            matches = [segment for segment in segments if np.dot(segment["normal"], sign) > .95]
            if matches:
                chosen.append(max(matches, key=lambda segment: segment["length"]))
        chosen_ids = {id(segment) for segment in chosen}
        for segment in sorted(segments, key=lambda value: (-value["length"], tuple(value["start"]))):
            if id(segment) not in chosen_ids:
                chosen.append(segment)
                chosen_ids.add(id(segment))
        face_candidates = chosen[:remaining]

    ordinary = []
    if face_candidates:
        allocation = np.ones(len(face_candidates), dtype=int)
        extra = remaining - len(face_candidates)
        lengths = np.asarray([segment["length"] for segment in face_candidates], dtype=float)
        if extra > 0:
            raw = extra * lengths / lengths.sum()
            allocation += np.floor(raw).astype(int)
            for index in np.argsort(-(raw - np.floor(raw)))[:extra - int(np.floor(raw).sum())]:
                allocation[int(index)] += 1
        for segment, amount in zip(face_candidates, allocation):
            for fraction in (np.arange(int(amount), dtype=float) + .5) / int(amount):
                ordinary.append({
                    "point": segment["start"] + fraction * (segment["end"] - segment["start"]),
                    "normal": segment["normal"],
                    "tangent": segment["tangent"],
                    "faces": np.stack([segment["normal"], np.zeros(2)]),
                    "offset": float(pusher_radius + clearance),
                    "type": "single",
                })
    modes = corners + ordinary
    if len(modes) < 8:
        raise ValueError(f"could only define {len(modes)} fixed modes")
    modes = modes[:8]
    points = np.asarray([mode["point"] for mode in modes], dtype=float)
    normals = np.asarray([mode["normal"] for mode in modes], dtype=float)
    tangents = np.asarray([mode["tangent"] for mode in modes], dtype=float)
    face_normals = np.zeros((8, 2, 2), dtype=float)
    face_mask = np.zeros((8, 2), dtype=bool)
    offsets = np.zeros(8, dtype=float)
    mode_types = []
    for index, mode in enumerate(modes):
        face_normals[index, 0] = mode["faces"][0]
        face_mask[index, 0] = True
        if mode["type"] == "corner_pair":
            face_normals[index, 1] = mode["faces"][1]
            face_mask[index, 1] = True
        offsets[index] = mode["offset"]
        mode_types.append(mode["type"])
    candidate_metadata = dict(metadata or {})
    candidate_metadata.update({
        "sampling": "fixed_exterior_faces_plus_concave_corner_pairs",
        "mode_count": 8,
        "pusher_radius": float(pusher_radius),
        "clearance": float(clearance),
        "mode_types": mode_types,
        "corner_candidate_count": len(corners),
    })
    return ModeLibrary(
        shape=shape,
        c=np.zeros((3, 8), dtype=float),
        contact_points=points,
        normals=normals,
        tangents=tangents,
        u_min=np.full(8, float(u_min), dtype=float),
        u_max=np.full(8, float(u_max), dtype=float),
        valid=np.ones(8, dtype=bool),
        face_normals=face_normals,
        face_mask=face_mask,
        contact_offset=offsets,
        mode_type=np.asarray(mode_types, dtype="U32"),
        metadata=candidate_metadata,
    )


def fit_mode_response(body_twists, scalar_inputs, weights=None):
    """Fit one raw ``c_i`` from measured body-frame twists.

    The scalar input is the commanded normal speed associated with each
    sample.  The fit is intentionally one-dimensional and keeps the raw
    angular response in rad/m rather than normalizing it away.
    """

    twists = np.asarray(body_twists, dtype=np.float64)
    inputs = np.asarray(scalar_inputs, dtype=np.float64).reshape(-1)
    if twists.ndim != 2 or twists.shape[1] != 3 or len(twists) != len(inputs):
        raise ValueError("body_twists must be (K,3) and match scalar_inputs")
    if len(inputs) < 1 or not np.isfinite(twists).all() or not np.isfinite(inputs).all():
        raise ValueError("calibration samples must be finite and nonempty")
    if np.any(inputs < 0):
        raise ValueError("scalar calibration inputs must be nonnegative")
    if weights is None:
        weights = np.ones(len(inputs), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != inputs.shape or np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and nonnegative")
    denominator = float(np.sum(weights * inputs * inputs))
    if denominator <= 1e-15:
        raise ValueError("calibration inputs have no nonzero excitation")
    response = np.sum((weights * inputs)[:, None] * twists, axis=0) / denominator
    residual = twists - inputs[:, None] * response[None, :]
    weight_sum = max(float(np.sum(weights)), 1e-12)
    scale = max(float(np.sqrt(np.sum(weights * np.sum(np.square(twists), axis=1)) / weight_sum)), 1e-12)
    relative_rmse = float(np.sqrt(np.sum(weights * np.sum(residual * residual, axis=1)) / weight_sum) / scale)
    return response, relative_rmse

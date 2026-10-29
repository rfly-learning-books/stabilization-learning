"""Versioned synthetic visual assets, independently built from PhysX truth boxes."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .runtime import activate_reference, json_hash, read_json, sha256, write_json


def shape_specs(catalog=None):
    activate_reference()
    from ca_glpr_unseen_shape.env.shapes import MIXED_SHAPES, ShapeSpec
    from ca_glpr_unseen_shape.env.shapes import ALL_SHAPE_NAMES
    specs = {name: MIXED_SHAPES[name] for name in ALL_SHAPE_NAMES}
    if catalog:
        data = read_json(catalog)
        if set(data) != {"shapes"}:
            raise ValueError("Custom catalogue must contain only a shapes mapping")
        for name, item in data["shapes"].items():
            if name in specs:
                raise ValueError(f"Custom shape would overwrite reference geometry: {name}")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid shape name: {name}")
            if set(item) != {"boxes", "orientation_period"}:
                raise ValueError("Each custom shape needs boxes and orientation_period")
            boxes = np.asarray(item["boxes"], dtype=float)
            period = float(item["orientation_period"])
            if boxes.ndim != 3 or boxes.shape[1:] != (2, 2) or not len(boxes):
                raise ValueError("boxes must be [ [[cx,cy],[half_x,half_y]], ... ]")
            if not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
                raise ValueError("Box dimensions must be finite and positive")
            if not np.isfinite(period) or period <= 0:
                raise ValueError("Specify a positive task orientation period")
            order = 2 * np.pi / period
            if not np.isclose(order, round(order)):
                raise ValueError("Orientation period must be 2pi divided by an integer")
            spec = ShapeSpec(name, tuple((tuple(c), tuple(s)) for c, s in boxes), period)
            if spec.radius() >= .24:
                raise ValueError("Truth shape does not fit the reference reset workspace")
            specs[name] = spec
    return specs


def spec_record(spec):
    return dict(name=spec.name, boxes=spec.boxes, orientation_period=float(spec.orientation_period))


def rasterize(spec, resolution=384, meters_per_pixel=.0015):
    x = (np.arange(resolution) + .5 - resolution / 2) * meters_per_pixel
    y = (resolution / 2 - np.arange(resolution) - .5) * meters_per_pixel
    xx, yy = np.meshgrid(x, y)
    mask = np.zeros_like(xx, dtype=bool)
    for (cx, cy), (hx, hy) in spec.boxes:
        mask |= (np.abs(xx - cx) <= hx) & (np.abs(yy - cy) <= hy)
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Raster canvas clips shape")
    return mask


def prepare_shape(spec, root):
    import cv2
    from scipy.ndimage import distance_transform_edt
    from shapely import box, unary_union
    activate_reference()
    from starter.build_mjx import build

    directory = Path(root) / spec.name
    request = dict(schema=1, truth=spec_record(spec), resolution=384,
                   meters_per_pixel=.0015, mass=.8, friction=3., thickness=.04,
                   frame="image", tolerance_mm=1.5)
    truth = unary_union([box(cx-hx, cy-hy, cx+hx, cy+hy) for (cx, cy), (hx, hy) in spec.boxes])
    # Reference Z has disconnected collision islands attached to ONE rigid body.
    # Allow this only when the independent truth specification explicitly has it.
    multipart = truth.geom_type == "MultiPolygon"
    if multipart:
        request["allow_disconnected"] = True
        request["truth_components"] = len(truth.geoms)
    fingerprint = json_hash(request)
    if directory.exists():
        geometry = Geometry(directory)
        if geometry.metadata["request_hash"] != fingerprint:
            raise ValueError(f"Changed geometry/config at {directory}; choose a new asset_root")
        return geometry
    mask = rasterize(spec, request["resolution"], request["meters_per_pixel"])
    directory.mkdir(parents=True, exist_ok=False)
    input_path = directory / "input_mask.png"
    if not cv2.imwrite(str(input_path), mask.astype(np.uint8) * 255):
        raise IOError("Could not write synthetic input mask")
    if multipart:
        from ca_glpr_unseen_shape.geometry.rgb_to_mjx import BuildConfig, from_mask, save_asset
        from starter.build_mjx import h1_xml
        import mujoco
        cfg = BuildConfig(request["meters_per_pixel"], thickness=.04, mass=.8,
                          tolerance_m=.0015, allow_disconnected=True)
        asset = from_mask(mask, cfg)
        if asset["metadata"]["components"] != request["truth_components"]:
            raise ValueError("Visual reconstruction changed the declared rigid-body topology")
        xml = h1_xml(asset, cfg, frame="image", object_friction=3.)
        native = mujoco.MjModel.from_xml_string(xml)
        if (native.nq, native.nv, native.nu) != (6, 6, 2):
            raise ValueError("Multipart asset must retain one object rigid body")
        save_asset(asset, directory / "model")
        (directory / "model/h1_model.xml").write_text(xml)
        meta = dict(format="dl_push_explicit_compound_h1_v1", horizon=1, control_dt=.05,
                    substeps=10, integrator_dt=.005, frame="image",
                    model_origin_in_image_m=[0., 0.], mass_assumed_kg=.8,
                    nominal_object_friction=3., nominal_table_friction=.5,
                    nominal_pusher_friction=1., reconstruction=asset["metadata"],
                    pusher_approximation="sphere_radius_0.018_not_PhysX_cylinder",
                    one_rigid_body=True, allow_disconnected=True,
                    xml_sha256=sha256(directory / "model/h1_model.xml"))
        write_json(directory / "model/h1_metadata.json", meta)
    else:
        meta = build(input_path, directory / "model", request["meters_per_pixel"], mask=True,
                     frame="image", thickness=.04, mass=.8, object_friction=3.)
    clean = cv2.imread(str(directory / "model/mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    sdf = (distance_transform_edt(~clean) - distance_transform_edt(clean)) * request["meters_per_pixel"]
    grid = np.stack([
        cv2.resize(clean.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA),
        cv2.resize(sdf.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA) / .10,
    ]).astype(np.float32)
    rows, cols = np.nonzero(clean)
    width = (cols.max() - cols.min() + 1) * request["meters_per_pixel"]
    height = (rows.max() - rows.min() + 1) * request["meters_per_pixel"]
    center = meta["reconstruction"]["image_centroid_world_m"]
    dimensions = np.array([width / .2, height / .2,
                           clean.sum() * request["meters_per_pixel"] ** 2 / .04,
                           .04 / .04, center[0] / .2, center[1] / .2], np.float32)
    np.savez_compressed(directory / "features.npz", grid=grid, sdf=sdf.astype(np.float32),
                        dimensions=dimensions)
    # PhysX still uses the original boxes, NOT the reconstructed collision meshes.
    metadata = dict(request=request, request_hash=fingerprint, shape=spec.name,
                    orientation_period=float(spec.orientation_period),
                    input_kind="calibrated_synthetic_mask_from_truth_boxes",
                    model_origin_xy_in_body=[0., 0.], model_yaw_offset=0.,
                    independent_physics_geometry=True,
                    nominal_contact_approximation=meta["pusher_approximation"],
                    reconstruction=meta["reconstruction"])
    metadata["files"] = {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*")) if p.is_file()
    }
    write_json(directory / "asset.json", metadata)
    return Geometry(directory)


class Geometry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.metadata = read_json(self.path / "asset.json")
        for name, digest in self.metadata["files"].items():
            if not (self.path / name).resolve().is_relative_to(self.path):
                raise ValueError("Unsafe asset path")
            if sha256(self.path / name) != digest:
                raise ValueError(f"Changed asset {self.path / name}")
        with np.load(self.path / "features.npz", allow_pickle=False) as data:
            self.grid = data["grid"].copy()
            self.sdf = data["sdf"].copy()
            self.dimensions = data["dimensions"].copy()
        self.mpp = self.metadata["request"]["meters_per_pixel"]
        self.name = self.metadata["shape"]
        self.period = self.metadata["orientation_period"]
        boxes = self.metadata["request"]["truth"]["boxes"]
        centers = np.asarray([item[0] for item in boxes], dtype=float)
        half_sizes = np.asarray([item[1] for item in boxes], dtype=float)
        # PhysX creates one collision shape per box.  Overlaps therefore
        # contribute to compound mass twice, unlike the visual polygon union.
        weights = np.prod(2 * half_sizes, axis=-1)
        self.physics_center_xy = np.average(centers, axis=0, weights=weights)
        self.model_origin_xy = np.asarray(
            self.metadata["model_origin_xy_in_body"], dtype=float
        )
        self.xml = self.path / "model/h1_model.xml"
        self.fingerprint = sha256(self.path / "asset.json")
        import shapely.geometry
        self.polygon = shapely.geometry.shape(read_json(self.path / "model/geometry.json")["polygon"])
        parts = (
            [self.polygon]
            if self.polygon.geom_type == "Polygon"
            else list(self.polygon.geoms)
        )
        samples = []
        total_length = sum(part.exterior.length for part in parts)
        for part in parts:
            count = max(
                16, int(np.ceil(128 * part.exterior.length / total_length))
            )
            for distance in np.linspace(
                0., part.exterior.length, count, endpoint=False
            ):
                samples.append(part.exterior.interpolate(distance).coords[0])
        self.contact_boundary_points = np.asarray(samples, dtype=np.float32)

    def local(self, state):
        """Nine body-frame SDF queries at/around the real pusher position."""
        from scipy.ndimage import map_coordinates
        angle = float(state[3])
        c, s = np.cos(angle), np.sin(angle)
        local = np.array([[c, s], [-s, c]]) @ (state[8:10] - state[:2])
        offsets = np.array([(x, y) for y in (-.018, 0., .018) for x in (-.018, 0., .018)])
        queries = local + offsets
        pixel = np.c_[self.sdf.shape[0] / 2 - queries[:, 1] / self.mpp - .5,
                      queries[:, 0] / self.mpp + self.sdf.shape[1] / 2 - .5]
        clipped = np.clip(pixel, [0, 0], np.array(self.sdf.shape) - 1)
        values = map_coordinates(self.sdf, clipped.T, order=1, mode="nearest")
        # Outside this padded canvas, extend the positive distance conservatively.
        values += np.linalg.norm(pixel - clipped, axis=-1) * self.mpp
        return values.astype(np.float32)

    def contact_gap(self, state):
        return float(self.local(state)[4] - .018)

    def contact_gaps(self, states):
        """Batch equivalent of the center SDF query in contact_gap."""
        from scipy.ndimage import map_coordinates
        states = np.asarray(states)
        shape = states.shape[:-1]
        x = states.reshape(-1, 12)
        yaw = x[:, 3].astype(float)
        c, s = np.cos(yaw), np.sin(yaw)
        delta = x[:, 8:10] - x[:, :2]
        local = np.column_stack([c*delta[:, 0]+s*delta[:, 1],
                                 -s*delta[:, 0]+c*delta[:, 1]])
        pixel = np.column_stack([self.sdf.shape[0]/2-local[:, 1]/self.mpp-.5,
                                 local[:, 0]/self.mpp+self.sdf.shape[1]/2-.5])
        clipped = np.clip(pixel, [0, 0], np.array(self.sdf.shape)-1)
        values = map_coordinates(self.sdf, clipped.T, order=1, mode="nearest")
        values += np.linalg.norm(pixel-clipped, axis=-1)*self.mpp
        return (values.astype(np.float32)-.018).reshape(shape)

    def contact_targets(self, states, goal, turn_gain=.15):
        """Vectorized useful pusher targets for terminal contact shaping."""
        x = np.asarray(states, dtype=np.float32)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        goal = np.asarray(goal, dtype=np.float32)
        direction = goal[None, :2] - x[:, :2]
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        fallback = np.array([np.cos(goal[2]), np.sin(goal[2])], np.float32)
        direction = np.where(norm > 1e-6, direction / np.maximum(norm, 1e-8), fallback)
        cosine, sine = np.cos(x[:, 3]), np.sin(x[:, 3])
        body_direction = np.stack(
            (
                cosine * direction[:, 0] + sine * direction[:, 1],
                -sine * direction[:, 0] + cosine * direction[:, 1],
            ),
            axis=-1,
        )
        query = -.4 * body_direction
        distances = np.square(
            query[:, None] - self.contact_boundary_points[None]
        ).sum(-1)
        boundary = self.contact_boundary_points[np.argmin(distances, axis=1)]
        local_target = boundary - .022 * body_direction
        order = 2 * np.pi / self.period
        angle_error = np.arctan2(
            np.sin(order * (goal[2] - x[:, 3])),
            np.cos(order * (goal[2] - x[:, 3])),
        ) / order
        tangent = np.stack((-direction[:, 1], direction[:, 0]), axis=-1)
        world_shift = -np.clip(
            turn_gain * angle_error, -.05, .05
        )[:, None] * tangent
        local_shift = np.stack(
            (
                cosine * world_shift[:, 0] + sine * world_shift[:, 1],
                -sine * world_shift[:, 0] + cosine * world_shift[:, 1],
            ),
            axis=-1,
        )
        local_target += local_shift
        world_target = x[:, :2] + np.stack(
            (
                cosine * local_target[:, 0] - sine * local_target[:, 1],
                sine * local_target[:, 0] + cosine * local_target[:, 1],
            ),
            axis=-1,
        )
        return world_target[0] if single else world_target

    def approach_target(self, state, goal, turn_gain=0.0):
        """Desired pusher center behind the object for translation/turning."""
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        direction = np.asarray(goal[:2]) - state[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([np.cos(goal[2]), np.sin(goal[2])])
        direction /= max(np.linalg.norm(direction), 1e-8)
        c, s = np.cos(state[3]), np.sin(state[3])
        rotation = np.array([[c, -s], [s, c]])
        behind = rotation.T @ (-direction * .4)
        boundary = np.asarray(nearest_points(self.polygon.boundary, Point(behind))[0].coords[0])
        target = state[:2] + rotation @ boundary - .022 * direction
        if turn_gain:
            from .state import wrap
            order = 2 * np.pi / self.period
            angle_error = wrap(order * (goal[2] - state[3])) / order
            tangent = np.array([-direction[1], direction[0]])
            target -= np.clip(turn_gain * angle_error, -.05, .05) * tangent
        return target.astype(np.float32)

    def approach_action(self, state, goal, turn_gain=0.0):
        """Finite startup heuristic only; it carries no CLF feasibility claim."""
        direction = np.asarray(goal[:2]) - state[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([np.cos(goal[2]), np.sin(goal[2])])
        direction /= max(np.linalg.norm(direction), 1e-8)
        target = self.approach_target(state, goal, turn_gain)
        velocity = 4.0 * (target - state[8:10])
        if np.linalg.norm(target - state[8:10]) < .025:
            velocity += .23 * direction
        return np.clip(velocity, -.30, .30).astype(np.float32)

    def contact_recovery_action(
        self, state, goal, clearance=.010, speed=.15, turn_gain=.06,
        target_local=None,
    ):
        """Move the pusher around a buffered convex hull toward a useful contact.

        This is a finite hybrid recovery heuristic.  It has its own step budget
        and is deliberately not labelled as satisfying the task CLF.
        """
        from shapely.geometry import LineString, Point
        from shapely.ops import nearest_points

        if clearance < 0 or not 0 < speed <= .30:
            raise ValueError("Invalid contact-recovery clearance or speed")
        state = np.asarray(state, dtype=float)
        goal = np.asarray(goal, dtype=float)
        yaw = float(state[3])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        local = rotation.T @ (state[8:10] - state[:2])
        if target_local is None:
            target_world = self.approach_target(state, goal, turn_gain)
            target = rotation.T @ (target_world - state[:2])
        else:
            target = np.asarray(target_local, dtype=float)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError("Recovery target must be a finite body-frame point")

        obstacle = self.polygon.buffer(
            .018 + clearance, join_style=2
        ).convex_hull
        point = Point(local)
        target_point = Point(target)
        if obstacle.contains(target_point):
            target = np.asarray(
                nearest_points(obstacle.boundary, target_point)[0].coords[0]
            )
            target_point = Point(target)

        # First detach from an existing or near contact instead of drawing a
        # straight segment through the object.
        if obstacle.contains(point) or point.distance(obstacle) < .001:
            boundary = np.asarray(
                nearest_points(obstacle.boundary, point)[0].coords[0]
            )
            outward = (
                boundary - local if obstacle.contains(point)
                else local - boundary
            )
            if np.linalg.norm(outward) < 1e-7:
                outward = boundary - np.asarray(obstacle.centroid.coords[0])
            outward /= max(np.linalg.norm(outward), 1e-8)
            waypoint = boundary + .002 * outward
        elif not LineString([local, target]).intersects(obstacle):
            waypoint = target
        else:
            ring = obstacle.exterior
            start_distance = ring.project(point)
            target_distance = ring.project(target_point)
            length = ring.length
            forward = (target_distance - start_distance) % length
            backward = length - forward
            signed_step = min(.03, forward) if forward <= backward else -min(.03, backward)
            waypoint = np.asarray(
                ring.interpolate((start_distance + signed_step) % length).coords[0]
            )
            outward = waypoint - np.asarray(obstacle.centroid.coords[0])
            waypoint += .002 * outward / max(np.linalg.norm(outward), 1e-8)

        waypoint_world = state[:2] + rotation @ waypoint
        velocity = 4.0 * (waypoint_world - state[8:10])
        norm = np.linalg.norm(velocity)
        if norm > speed:
            velocity *= speed / norm
        return np.clip(velocity, -.30, .30).astype(np.float32)

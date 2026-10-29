"""Geometry-only release/transit/engage transfer graph construction."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass
class TransferGraph:
    distance: np.ndarray
    time: np.ndarray
    valid: np.ndarray
    paths: list[list[list[list[float]]]]
    metadata: dict

    def __post_init__(self):
        self.distance = np.asarray(self.distance, dtype=np.float64)
        self.time = np.asarray(self.time, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.distance.ndim != 2 or self.distance.shape[0] != self.distance.shape[1]:
            raise ValueError("distance must be square")
        if self.time.shape != self.distance.shape or self.valid.shape != self.distance.shape:
            raise ValueError("transfer matrices must have the same square shape")
        if len(self.paths) != self.distance.shape[0]:
            raise ValueError("paths must have one source row per mode")


def _components(polygon):
    return [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)


def _visibility_path(start, goal, obstacle, workspace, tolerance=1e-9):
    """Shortest path through a polygonal free space using visibility edges."""

    from shapely.geometry import LineString, Point, Polygon, MultiPolygon

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (2,) or goal.shape != (2,):
        raise ValueError("transfer endpoints must be 2-D")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        return None
    box = workspace
    if not box.contains(Point(start)) and not box.touches(Point(start)):
        return None
    if not box.contains(Point(goal)) and not box.touches(Point(goal)):
        return None
    direct = LineString([start, goal])
    if not direct.crosses(obstacle) and not direct.within(obstacle):
        return [start.tolist(), goal.tolist()]

    nodes = [start, goal]
    for polygon in _components(obstacle):
        for x, y in polygon.exterior.coords[:-1]:
            nodes.append(np.array([x, y], dtype=float))
    graph = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            segment = LineString([nodes[i], nodes[j]])
            if segment.length <= tolerance:
                continue
            # A segment may touch a boundary, but cannot enter the inflated
            # obstacle interior.  Tiny numerical endpoint touches are okay.
            if segment.crosses(obstacle) or segment.within(obstacle):
                continue
            if not box.contains(segment) and not box.touches(segment):
                continue
            weight = float(segment.length)
            graph[i].append((j, weight))
            graph[j].append((i, weight))
    distances = [math.inf] * len(nodes)
    previous = [None] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, index = heapq.heappop(queue)
        if distance > distances[index] + tolerance:
            continue
        if index == 1:
            break
        for neighbor, weight in graph[index]:
            candidate = distance + weight
            if candidate < distances[neighbor] - tolerance:
                distances[neighbor] = candidate
                previous[neighbor] = index
                heapq.heappush(queue, (candidate, neighbor))
    if not np.isfinite(distances[1]):
        return None
    chain = []
    index = 1
    while index is not None:
        chain.append(nodes[index].tolist())
        index = previous[index]
    return list(reversed(chain))


def build_transfer_graph(
    polygon,
    contact_points,
    normals,
    pusher_radius=0.018,
    clearance=0.002,
    workspace_limit=0.232,
    free_speed=0.20,
    engage_speed=0.01,
):
    """Build a real-path-counted transfer graph for sampled modes.

    The graph is conservative: every mode changes from its contact center to a
    release point, crosses the inflated geometry in free space, then engages
    the target.  If no collision-checked path exists, the edge is invalid.
    """

    from shapely.geometry import box

    points = np.asarray(contact_points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape:
        raise ValueError("contact_points and normals must both be (N,2)")
    if len(points) < 1 or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("transfer inputs must be finite and nonempty")
    if min(pusher_radius, clearance, free_speed, engage_speed) <= 0:
        raise ValueError("transfer scales must be positive")
    obstacle = polygon.buffer(float(pusher_radius) + float(clearance), join_style=2)
    workspace = box(-float(workspace_limit), -float(workspace_limit),
                    float(workspace_limit), float(workspace_limit))
    centers = points + normals * (float(pusher_radius) + float(clearance))
    release = points + normals * (float(pusher_radius) + 2.0 * float(clearance))
    count = len(points)
    distance = np.full((count, count), np.inf, dtype=float)
    times = np.full((count, count), np.inf, dtype=float)
    valid = np.zeros((count, count), dtype=bool)
    paths = [[[] for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(count):
            if i == j:
                distance[i, j] = 0.0
                times[i, j] = 0.0
                valid[i, j] = True
                paths[i][j] = [centers[i].tolist()]
                continue
            first = _visibility_path(centers[i], release[i], obstacle, workspace)
            middle = _visibility_path(release[i], release[j], obstacle, workspace)
            last = _visibility_path(release[j], centers[j], obstacle, workspace)
            if first is None or middle is None or last is None:
                continue
            chain = first + middle[1:] + last[1:]
            lengths = [
                float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
                for a, b in zip(chain[:-1], chain[1:])
            ]
            path_distance = float(np.sum(lengths))
            distance[i, j] = path_distance
            times[i, j] = path_distance / float(free_speed) + 2.0 * float(clearance) / float(engage_speed)
            valid[i, j] = True
            paths[i][j] = chain
    return TransferGraph(
        distance=distance,
        time=times,
        valid=valid,
        paths=paths,
        metadata=dict(
            pusher_radius=float(pusher_radius),
            clearance=float(clearance),
            workspace_limit=float(workspace_limit),
            free_speed=float(free_speed),
            engage_speed=float(engage_speed),
            obstacle_buffer=float(pusher_radius + clearance),
        ),
    )


"""Book-order collection using the retained physical RGB episode sampler."""
from __future__ import annotations

from copy import deepcopy
import json
import time

import numpy as np


def _write_record(path, record):
    """Commit a completed collection record without replacing prior evidence."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.rename(path)


def book_collect(backend, initial=False):
    """Import acknowledgment or one original-order round of new trajectories.

    ``initialize`` has already copied and verified the bootstrap replay and
    restored its RNG.  Collection IDs are zero for that import and one-based
    for the 24 subsequent rounds.  Network updates occur outside this method.
    """
    backend.guard()
    if not backend.initialized:
        raise RuntimeError("Initialize the book backend before collection")
    started = time.perf_counter()
    collection = int(backend.collection_id)
    before = int(backend.replay.transitions)
    if initial:
        if collection != 0 or not backend.replay.rows:
            raise RuntimeError("Initial collection requires the imported bootstrap replay")
        if any(row["split"] != "train" for row in backend.replay.rows):
            raise ValueError("Bootstrap replay contains non-training rows")
        result = dict(schema="rgb_book_collection_v1", stage="initial_collection",
                      collection=0, fresh_transitions=0,
                      imported_transitions=before, replay_transitions=before,
                      prior_replay_transitions=before, old_replay_loaded=True,
                      scene_count=len({r["scene_id"] for r in backend.replay.rows}),
                      episodes=len(backend.replay.rows), coverage=[],
                      wall_s=time.perf_counter()-started,
                      source="reconstructed_visual_bootstrap_20261023_v1")
        backend.collection_id = 1
        if getattr(backend, "fixed_probe", None) is None and hasattr(backend, "_record_probe"):
            replay_rng = deepcopy(backend.replay.rng.bit_generator.state)
            try:
                backend.fixed_probe = backend.replay.sample(int(backend.config.get("probe_batch_size", 4)))
            finally:
                backend.replay.rng.bit_generator.state = replay_rng
            backend._record_probe("initial_data")
        _write_record(backend.output / "RGB_COLLECTION_000.json", result)
        backend.event("rgb_book_bootstrap_imported", **result)
        return result

    if collection < 1:
        raise RuntimeError("Acknowledge the imported initial replay before collecting a round")

    from lac_vs.environment import sample_start
    from lac_vs import legacy
    import train_goal_fixed as joint
    from stabilization_gym.envs.tasks.rgb_book_physical import create_book_room

    round_index = collection - 1
    cfg = backend.cfg
    horizon = int(cfg["collection"]["steps"])
    per_scene = int(cfg["collection"]["episodes_per_scene_per_round"])
    if horizon <= 0 or per_scene <= 0:
        raise ValueError("Positive book collection horizon and episode count required")
    if any(scene["split"] != "train" for scene in backend.train_scenes):
        raise ValueError("Only training rooms may enter collection")

    # Do not sort, interleave or batch this list: its order determines the old
    # goal/difficulty schedule and the sequence of random draws.
    todo = [(scene, k) for scene in backend.train_scenes for k in range(per_scene)]
    coverage = []
    fresh = 0
    with (backend.output / "book_collection.jsonl").open("a", encoding="utf-8") as stream:
        for index, (scene, episode_index) in enumerate(todo):
            backend.guard()
            instance = create_book_room(cfg, scene)
            try:
                goal_id = (round_index + episode_index) % len(scene["refs"])
                frame = legacy.module(cfg, "room_env").ReferenceFrame(
                    np.array(scene["refs"][goal_id]["goal_matrix"]))
                difficulty = ("small", "medium", "large")[(round_index + index) % 3]
                offset = sample_start(cfg, scene, frame, backend.rng, difficulty)
                arrays, row = joint.episode(cfg, backend.output, scene, goal_id, offset,
                                            backend.learner, backend.depth, backend.rng,
                                            steps=horizon, collect=True, inst=instance)
                backend.replay.add(arrays, row,
                                   f"derivative_round{round_index:02d}_{index:04d}")
            finally:
                instance.close()
            length = len(arrays["action"])
            fresh += length
            record = dict(phase="student", round=round_index, index=index,
                          difficulty=difficulty, **row)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            coverage.append(dict(scene_id=scene["id"], goal_id=goal_id,
                                 difficulty=difficulty, episode=episode_index,
                                 transitions=length, status=row["status"],
                                 offset_start=row["offset_start"]))
            backend.event("rgb_book_collection_episode", collection=collection,
                          round=round_index, index=index+1, planned=len(todo),
                          scene_id=scene["id"], transitions=length,
                          fresh_transitions=fresh, status=row["status"])

    result = dict(schema="rgb_book_collection_v1", stage="collection",
                  collection=collection, round=round_index, fresh_transitions=fresh,
                  replay_transitions=int(backend.replay.transitions),
                  prior_replay_transitions=before, old_replay_loaded=True,
                  scene_count=len(backend.train_scenes), episodes=len(todo),
                  maximum_episode_steps=horizon, post_success_steps=0,
                  coverage=coverage, wall_s=time.perf_counter()-started,
                  sampling="original manifest scene order; scene then episode",
                  termination="collision, workspace exit, 11 consecutive successful observations, or horizon")
    _write_record(backend.output / f"RGB_COLLECTION_{collection:03d}.json", result)
    backend.collection_id += 1
    return result

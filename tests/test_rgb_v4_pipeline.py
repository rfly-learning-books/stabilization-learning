from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/"stabilization-gym", ROOT/"stabilization-learning-control"):
    sys.path.insert(0, str(folder))
from extensions.joint_certificate_v4.replay import StratifiedRGBReplay
from extensions.joint_certificate_v4.config import RGBConfig
from extensions.joint_certificate_v4.rollout import diagnostics


def episode(n=2):
    a = dict(rgb=np.zeros((n+1, 4, 4, 3), dtype=np.uint8), goal_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
             offset=np.zeros((n+1, 4)), rotation=np.repeat(np.eye(3)[None], n+1, axis=0),
             dt=np.ones(n)*.1, action=np.zeros((n, 4)), applied_action=np.zeros((n, 4)), previous_action=np.zeros((n, 4)),
             controller=np.zeros(n, dtype=bool), terminated=np.zeros(n, dtype=bool), truncated=np.zeros(n, dtype=bool))
    a["truncated"][-1] = True
    return a


def test_all_archive_remains_sampleable_after_mapping_eviction(tmp_path):
    replay = StratifiedRGBReplay(tmp_path/"archive", np.random.default_rng(1), mmap_episodes=1)
    try:
        for i in range(10):
            replay.add(episode(), dict(split="train", stratum=str(i%2), behavior_version=i))
            replay._load(i)
        assert len(replay.cache) == 1
        assert len(replay.rows) == 10
        keys = replay.sample_keys(2000)
        assert set(i for i, _ in keys) == set(range(10))
        assert sum(replay.rows[i]["stratum"] == "0" for i, _ in keys) == 1000
        assert replay.snapshot()["discarded_episodes"] == 0
        assert replay._load(0)["rgb"].shape[0] == 3
    finally:
        replay.close()


def test_replay_refuses_reset_leakage_and_nontraining(tmp_path):
    replay = StratifiedRGBReplay(tmp_path/"archive", np.random.default_rng(1))
    a = episode()
    with pytest.raises(ValueError, match="training"):
        replay.add(a, dict(split="test"))
    a["dt"][0] = 0
    with pytest.raises(ValueError, match="Reset"):
        replay.add(a, dict(split="train"))
    a = episode(); a["terminated"][0] = True
    with pytest.raises(ValueError, match="boundary"):
        replay.add(a, dict(split="train"))


def test_formal_scope_and_coverage():
    config = RGBConfig()
    scenes = json.loads((ROOT/"migration/visual_scene_reconstruction/scenes.json").read_text(encoding="utf8"))["scenes"]
    strata = sum(len(s["refs"])*4 for s in scenes if s["split"] == "train")
    assert strata == 384
    assert sum(config.transitions_per_stratum)*strata == 129024
    assert config.calibration_samples == strata
    assert config.warmup_updates+2*config.joint_pairs == 14000
    with pytest.raises(ValueError, match="every declared"):
        RGBConfig(test_case_limit=16)
    with pytest.raises(ValueError):
        RGBConfig(collection_pairs=(0, 6500))
    assert RGBConfig.from_dict(__import__("dataclasses").asdict(config)) == config


def test_diagnostics_retains_value_increase_and_wrong_direction():
    a = dict(V=np.array([.5, 2., .5]), physical_error=np.ones(3), dt=np.ones(2)*.1,
             J=np.array([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]),
             error=np.array([[1.], [2.], [1.]]), applied_action=np.ones((2, 4)))
    result = diagnostics(a, RGBConfig().learner)
    assert result["actual_V_increase_fraction"] == .5
    assert result["J_opposite_fraction"] == .5
    assert result["J_effective_steps"] == 2


def test_strided_rgb_archive_is_exact(tmp_path):
    from extensions.joint_certificate_v4.replay import write_array
    import hashlib
    vector = np.random.default_rng(8).integers(0, 256, (5, 4, 16, 16, 3), dtype=np.uint8)
    lane = vector[:, 2]
    assert not lane.flags.c_contiguous
    fast, reference = tmp_path/"fast.npy", tmp_path/"reference.npy"
    write_array(fast, lane)
    np.save(reference, lane, allow_pickle=False)
    assert hashlib.sha256(fast.read_bytes()).digest() == hashlib.sha256(reference.read_bytes()).digest()


def test_owned_process_tree_exit(tmp_path):
    import os
    import subprocess
    import time
    import psutil
    from extensions.joint_certificate_v4.process_control import terminate_owned_tree
    record = tmp_path/"pids.json"
    script = tmp_path/"sleeper.py"
    script.write_text("import subprocess,sys,os,json,time\n"
                      "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
                      "open(sys.argv[1],'w').write(json.dumps([os.getpid(),child.pid]))\n"
                      "time.sleep(60)\n", encoding="utf8")
    process = subprocess.Popen([sys.executable, str(script), str(record)],
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                              start_new_session=os.name != "nt")
    try:
        until = time.monotonic()+15
        while not record.exists() and time.monotonic() < until:
            time.sleep(.05)
        pids = json.loads(record.read_text())
        descendants = psutil.Process(process.pid).children(recursive=True)
        assert all(psutil.pid_exists(p) for p in pids)
        terminate_owned_tree(process)
        _, alive = psutil.wait_procs(descendants, timeout=5)
        assert not alive
        assert process.poll() is not None
    finally:
        terminate_owned_tree(process)

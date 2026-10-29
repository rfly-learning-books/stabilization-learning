"""Supervisor control-flow checks with fake processes; no training is launched."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from stabilization_learning.joint_v2.runtime import RunStopped, atomic_json


@pytest.fixture
def harness(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("joint_v2_night_under_test", directory / "run_joint_certificate_night_v2.py")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    original_guard = supervisor.stage_guard
    root = tmp_path / "night"
    root.mkdir()
    atomic_json(root / "NIGHT_PROTOCOL.json", {"seeds": [81, 82], "time_bank": str(tmp_path / "bank.json")})
    clock = [1000.]
    deadline = 1000. + 14 * 3600
    state = SimpleNamespace(module=supervisor, root=root, clock=clock, deadline=deadline,
                            commands=[], exits={}, long_stage=None, stopped=[], guard_calls=[],
                            original_guard=original_guard, fail_spawn=None, finish_after_deadline=None)

    def guarded(args):
        state.guard_calls.append(args)
        def check():
            if clock[0] >= deadline:
                raise RunStopped("Original shared deadline reached")
            if (root / "STOP").exists():
                raise RunStopped("STOP present")
        check()
        return check, {"started_unix": clock[0], "deadline_unix": deadline,
                       "original_deadline_utc": "test-original-deadline"}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command, self.kwargs = list(command), kwargs
            self.stage = Path(kwargs["stdout"].name).stem
            if state.fail_spawn == self.stage:
                raise OSError("Simulated launch failure")
            self.pid = 40000 + len(state.commands)
            self.returncode = None
            state.commands.append(self)
        def poll(self):
            if self.returncode is None and self.stage != state.long_stage:
                self.returncode = state.exits.get(self.stage, 0)
                if self.stage == state.finish_after_deadline:
                    clock[0] = deadline + 0.01
            return self.returncode
        def kill(self):
            self.returncode = -9
        def wait(self, timeout=None):
            if self.returncode is None:
                raise supervisor.subprocess.TimeoutExpired(self.command, timeout)
            return self.returncode

    def stopped(process):
        if process.poll() is not None:
            return
        state.stopped.append(process.pid)
        process.kill()

    def unexpected_process(*args, **kwargs):
        raise AssertionError("The test must never launch a real subprocess")

    monkeypatch.setattr(supervisor, "stage_guard", guarded)
    monkeypatch.setattr(supervisor, "stop_owned_child", stopped)
    monkeypatch.setattr(supervisor.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(supervisor.subprocess, "run", unexpected_process)
    monkeypatch.setattr(supervisor.time, "time", lambda: clock[0])
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return state


def stages(harness):
    return [process.stage for process in harness.commands]


def test_no_plan_creation_or_process_launch_on_import(harness, monkeypatch, tmp_path):
    assert harness.commands == []
    missing = tmp_path / "missing-plan"
    monkeypatch.setattr(harness.module, "stage_guard", harness.original_guard)
    with pytest.raises(FileNotFoundError):
        harness.module.supervise(missing)
    assert not missing.exists() and harness.commands == []


def test_command_plan_reuses_one_night_and_has_no_init_or_renewed_budget(harness):
    plan = harness.module.command_plan(harness.root, [81, 82])
    commands = [command for group in plan.values() for _, command in group]
    for command in commands:
        assert "init-night" not in command and "--max-hours" not in command
        if "verify" not in command:
            assert command[command.index("--night-root") + 1] == str(harness.root.resolve())
    assert len(plan["rgb_train"]) == len(plan["push_train"]) == 2


def test_successful_execution_uses_one_guard_and_does_not_claim_scientific_success(harness):
    result = harness.module.supervise(harness.root)
    assert result["status"] == "complete"
    assert result["deadline_unix"] == harness.deadline
    assert result["scientific_analysis_complete"] is False
    assert len(harness.guard_calls) == 1
    assert stages(harness) == ["verify", "rgb_train_seed81", "rgb_train_seed82", "rgb_group_test",
                              "push_train_seed81", "push_train_seed82", "push_group_freeze",
                              "push_test_seed81", "push_test_seed82", "arm_seed81", "arm_seed82"]
    assert all(process.kwargs["stdin"] == harness.module.subprocess.DEVNULL for process in harness.commands)
    assert not (harness.root / "NIGHT_WRITER.lock").exists()
    assert (harness.root / "FINAL_STATE.json").exists()


def test_failed_rgb_seed_cannot_test_or_retrain_its_group(harness):
    harness.exits["rgb_train_seed81"] = 101
    result = harness.module.supervise(harness.root)
    names = stages(harness)
    assert result["status"] == "partial_failed"
    assert names.count("rgb_train_seed81") == 1
    assert "rgb_train_seed82" not in names and "rgb_group_test" not in names
    assert "push_train_seed81" in names and "arm_seed82" in names


@pytest.mark.parametrize("failure", ["push_train_seed82", "push_group_freeze"])
def test_incomplete_push_group_never_tests_or_deploys(harness, failure):
    harness.exits[failure] = 2
    result = harness.module.supervise(harness.root)
    names = stages(harness)
    assert result["status"] == "partial_failed"
    assert names.count(failure) == 1
    assert not any(name.startswith(("push_test_", "arm_")) for name in names)
    if failure.startswith("push_train"):
        assert "push_group_freeze" not in names
    assert "rgb_group_test" in names


def test_failed_engineering_verification_launches_no_training(harness):
    harness.exits["verify"] = 1
    result = harness.module.supervise(harness.root)
    assert result["status"] == "failed" and stages(harness) == ["verify"]
    assert "verification failed" in result["detail"]


def test_supervisor_never_resumes_a_completed_or_interrupted_night(harness):
    harness.exits["verify"] = 1
    harness.module.supervise(harness.root)
    before = len(harness.commands)
    original = (harness.root / "FINAL_STATE.json").read_bytes()
    with pytest.raises(FileExistsError, match="automatic resume"):
        harness.module.supervise(harness.root)
    assert len(harness.commands) == before
    assert (harness.root / "FINAL_STATE.json").read_bytes() == original


def test_hard_deadline_stops_only_current_owned_child_and_no_new_stage(harness):
    harness.clock[0] = harness.deadline - 0.2
    harness.long_stage = "rgb_train_seed81"
    result = harness.module.supervise(harness.root)
    assert result["status"] == "stopped" and "deadline" in result["detail"]
    assert stages(harness) == ["verify", "rgb_train_seed81"]
    assert harness.stopped == [harness.commands[-1].pid]
    assert harness.commands[0].returncode == 0
    assert result["deadline_unix"] == harness.deadline
    assert not (harness.root / "NIGHT_WRITER.lock").exists()


def test_final_stage_finishing_after_deadline_cannot_be_called_complete(harness):
    harness.finish_after_deadline = "arm_seed82"
    result = harness.module.supervise(harness.root)
    assert stages(harness)[-1] == "arm_seed82"
    assert result["status"] == "stopped" and "deadline" in result["detail"]
    # The process already exited; do not terminate any finished or unrelated PID.
    assert harness.stopped == []


def test_stage_stop_preserves_marker_and_stops_owned_process(harness):
    stop = harness.root / "rgb_seed81" / "STOP"
    stop.parent.mkdir()
    stop.write_text("keep evidence", encoding="utf-8")
    harness.long_stage = "verify"
    result = harness.module.supervise(harness.root)
    assert result["status"] == "stopped" and "STOP" in result["detail"]
    assert stages(harness) == ["verify"] and harness.stopped == [harness.commands[0].pid]
    assert stop.read_text(encoding="utf-8") == "keep evidence"


def test_launch_failure_is_recorded_without_restarting(harness):
    harness.fail_spawn = "rgb_train_seed81"
    result = harness.module.supervise(harness.root)
    assert result["status"] == "failed" and "launch failure" in result["detail"]
    assert stages(harness) == ["verify"]
    assert not (harness.root / "NIGHT_WRITER.lock").exists()


def test_windows_termination_targets_only_owned_pid(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("joint_v2_stop_under_test", directory / "run_joint_certificate_night_v2.py")
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    calls = []
    process = SimpleNamespace(pid=12345, poll=lambda: None, wait=lambda timeout: 0,
                              kill=lambda: pytest.fail("taskkill already completed"))
    monkeypatch.setattr(supervisor, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(supervisor.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    supervisor.stop_owned_child(process)
    assert calls[0][0] == ["taskkill", "/PID", "12345", "/T", "/F"]
    assert calls[0][1]["stdin"] == supervisor.subprocess.DEVNULL
    assert len(calls) == 1
    calls.clear()
    process.poll = lambda: 0
    supervisor.stop_owned_child(process)
    assert calls == []

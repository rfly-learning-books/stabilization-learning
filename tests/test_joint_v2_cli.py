"""Root command safety without launching a trainer or touching formal runs."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stabilization_learning.joint_v2.runtime import RunStopped, atomic_json, file_sha256


@pytest.fixture
def cli(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_joint_certificate_v2.py"
    spec = importlib.util.spec_from_file_location("joint_v2_cli_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A harmless source is frozen so other developers can edit the CLI while
    # these unit tests are running; the source-change check itself stays real.
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "sources", lambda: {str(source): file_sha256(source)})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    return module


@pytest.fixture
def night(cli, monkeypatch, tmp_path):
    now = [1_800_000_000.]
    monkeypatch.setattr(cli.time, "time", lambda: now[0])
    bank = tmp_path / "bank.json"
    bank.write_text('{"fixture": "configuration-only bank; never loaded by a trainer"}', encoding="utf-8")
    root = tmp_path / "night"
    args = cli.build_parser().parse_args([
        "init-night", "--output", str(root), "--time-bank", str(bank),
        "--seeds", "81", "82", "--max-hours", "14",
    ])
    protocol = cli.init_night(args)
    return SimpleNamespace(root=root, bank=bank, protocol=protocol, clock=now, args=args)


def test_init_night_generates_only_configuration_and_never_launches(cli, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Configuration generation may not start any process or trainer")
    import extensions.joint_certificate_v2.rgb as rgb
    import stabilization_learning.joint_v2.push as push
    # Windows platform discovery may invoke `ver`; it is metadata, not a
    # trainer. Resolve it first, then forbid launches during plan generation.
    fingerprint = cli.runtime_fingerprint()
    monkeypatch.setattr(cli, "runtime_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(cli.subprocess, "run", forbidden)
    monkeypatch.setattr(rgb, "train", forbidden)
    monkeypatch.setattr(push, "train", forbidden)
    bank = tmp_path / "bank.json"
    bank.write_text("{}", encoding="utf-8")
    root = tmp_path / "night"
    assert cli.main(["init-night", "--output", str(root), "--time-bank", str(bank),
                     "--seeds", "81", "82", "--max-hours", "14"]) == 0
    protocol = cli.read(root / "NIGHT_PROTOCOL.json")
    assert protocol["deadline_unix"] - protocol["started_unix"] == 14 * 3600
    assert protocol["time_bank_sha256"] == file_sha256(bank)
    assert not list(root.rglob("*.pt")) and not list(root.rglob("WRITER.lock"))
    assert not list(root.rglob("FINAL_STATE.json"))
    for seed in (81, 82):
        config = rgb.RGBRunConfig.from_dict(cli.read(root / "configs" / f"rgb_seed{seed}.json"))
        assert config.seed == seed and set(config.group_seeds) == {81, 82}
        assert config.learner is not None
        push_config = cli.read(root / "configs" / f"push_seed{seed}.json")
        assert push_config["seed"] == seed


@pytest.mark.parametrize("hours", [0., -1., 14.01, float("nan"), float("inf")])
def test_init_rejects_invalid_budget_without_claiming_output(cli, tmp_path, hours):
    root = tmp_path / "invalid"
    bank = tmp_path / "bank.json"
    bank.write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(["init-night", "--output", str(root), "--time-bank", str(bank)])
    args.max_hours = hours
    with pytest.raises(ValueError):
        cli.init_night(args)
    assert not root.exists()


@pytest.mark.parametrize("seeds", [[], [81, 81]])
def test_init_rejects_empty_or_duplicate_groups(cli, tmp_path, seeds):
    root = tmp_path / "invalid"
    bank = tmp_path / "bank.json"
    bank.write_text("{}", encoding="utf-8")
    args = cli.build_parser().parse_args(["init-night", "--output", str(root), "--time-bank", str(bank)])
    args.seeds = seeds
    with pytest.raises(ValueError):
        cli.init_night(args)
    assert not root.exists()


def test_init_never_overwrites_existing_night(cli, night):
    original = (night.root / "NIGHT_PROTOCOL.json").read_bytes()
    with pytest.raises(FileExistsError):
        cli.init_night(night.args)
    assert (night.root / "NIGHT_PROTOCOL.json").read_bytes() == original


def test_every_stage_shares_original_fourteen_hour_deadline(cli, night):
    args = SimpleNamespace(night_root=str(night.root), max_hours=99., output=str(night.root / "stage"))
    night.clock[0] += 2 * 3600
    guard1, limits1 = cli.stage_guard(args)
    night.clock[0] += 4 * 3600
    guard2, limits2 = cli.stage_guard(args)
    assert limits1["deadline_unix"] == limits2["deadline_unix"] == night.protocol["deadline_unix"]
    assert limits2["deadline_unix"] - night.clock[0] == 8 * 3600
    night.clock[0] = night.protocol["deadline_unix"]
    for guard in (guard1, guard2):
        with pytest.raises(RunStopped, match="deadline"):
            guard()
    with pytest.raises(RunStopped, match="deadline"):
        cli.stage_guard(args)


@pytest.mark.parametrize("stop_location", ["night", "stage"])
def test_stop_file_is_obeyed_without_deleting_it(cli, night, stop_location):
    output = night.root / "stage"
    output.mkdir()
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(output))
    guard, _ = cli.stage_guard(args)
    stop = (night.root if stop_location == "night" else output) / "STOP"
    stop.write_text("keep diagnostic stop", encoding="utf-8")
    with pytest.raises(RunStopped, match="STOP"):
        guard()
    assert stop.read_text(encoding="utf-8") == "keep diagnostic stop"


def test_changed_frozen_source_is_rejected_before_stage(cli, night):
    source = Path(next(iter(night.protocol["sources"])))
    source.write_text("VERSION = 2\n", encoding="utf-8")
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(night.root / "stage"))
    with pytest.raises(RuntimeError, match="source"):
        cli.stage_guard(args)


@pytest.mark.parametrize("target", ["config", "bank"])
def test_frozen_config_and_bank_tampering_rejected_before_stage(cli, night, target):
    path = night.root / "configs" / "rgb_seed81.json" if target == "config" else night.bank
    path.write_text(path.read_text(encoding="utf-8") + "\n ", encoding="utf-8")
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(night.root / "stage"))
    with pytest.raises(RuntimeError, match=target):
        cli.stage_guard(args)


def test_protocol_seal_and_running_protocol_guard(cli, night):
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(night.root / "stage"))
    guard, _ = cli.stage_guard(args)
    path = night.root / "NIGHT_PROTOCOL.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RunStopped, match="protocol"):
        guard()
    with pytest.raises(RuntimeError, match="seal"):
        cli.stage_guard(args)


@pytest.mark.parametrize("change", [
    {"deadline_unix": float("nan")},
    {"max_hours": 15.},
    {"max_hours": -1.},
    {"deadline_unix": 1_800_000_000. + 15 * 3600},
])
def test_protocol_rejects_invalid_budget_even_with_matching_seal(cli, night, change):
    protocol = dict(night.protocol, **change)
    path = night.root / "NIGHT_PROTOCOL.json"
    # Intentionally malformed JSON numeric value checks defensive parsing.
    path.write_text(json.dumps(protocol), encoding="utf-8")
    (night.root / "NIGHT_PROTOCOL.sha256").write_text(file_sha256(path) + "\n", encoding="utf-8")
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(night.root / "stage"))
    with pytest.raises(ValueError, match="budget"):
        cli.stage_guard(args)


def test_formal_output_cannot_escape_its_night_root(cli, night, tmp_path):
    args = SimpleNamespace(night_root=str(night.root), max_hours=None, output=str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="inside"):
        cli.stage_guard(args)


def test_stage_must_use_exact_frozen_config_or_identical_copy(cli, night, tmp_path):
    planned = night.root / "configs" / "rgb_seed81.json"
    copied = tmp_path / "rgb.json"
    copied.write_bytes(planned.read_bytes())
    args = SimpleNamespace(night_root=str(night.root), config=str(copied))
    cli.require_planned_config(args, planned.name)
    config = cli.read(copied)
    config["joint_pairs"] += 1
    atomic_json(copied, config)
    with pytest.raises(ValueError, match="frozen night config"):
        cli.require_planned_config(args, planned.name)
    args.config = None
    with pytest.raises(ValueError, match="frozen night config"):
        cli.require_planned_config(args, planned.name)


def test_preflight_templates_are_frozen_with_declared_seed_group(cli, tmp_path):
    from extensions.joint_certificate_v2.rgb import RGBRunConfig
    from stabilization_learning.joint_v2.push import PushTrainConfig
    rgb = tmp_path / "rgb_template.json"
    push = tmp_path / "push_template.json"
    costs = tmp_path / "costs_template.json"
    atomic_json(rgb, asdict(RGBRunConfig(seed=99, group_seeds=(99,), device="cpu", joint_pairs=3)))
    atomic_json(push, asdict(PushTrainConfig(seed=99, updates=7)))
    atomic_json(costs, {"time": 2.})
    bank = tmp_path / "bank.json"
    bank.write_text("{}", encoding="utf-8")
    root = tmp_path / "templated_night"
    args = cli.build_parser().parse_args([
        "init-night", "--output", str(root), "--time-bank", str(bank), "--seeds", "81", "82",
        "--rgb-config", str(rgb), "--push-config", str(push), "--costs", str(costs),
    ])
    protocol = cli.init_night(args)
    for seed in (81, 82):
        generated = cli.read(root / "configs" / f"rgb_seed{seed}.json")
        assert generated["seed"] == seed and generated["group_seeds"] == [81, 82]
        assert generated["device"] == "cpu" and generated["joint_pairs"] == 3
        generated_push = cli.read(root / "configs" / f"push_seed{seed}.json")
        assert generated_push["seed"] == seed and generated_push["updates"] == 7
    assert cli.read(root / "configs" / "push_costs.json")["time"] == 2.
    assert len(protocol["configs"]) == 5
    assert all(file_sha256(path) == digest for path, digest in protocol["configs"].items())


def _planned_group(cli, night, kind):
    roots = []
    for seed in night.protocol["seeds"]:
        root = night.root / f"{kind}_seed{seed}"
        root.mkdir()
        protocol = {"config": cli.read(night.root / "configs" / f"{kind}_seed{seed}.json")}
        if kind == "rgb":
            protocol["deadline_unix"] = night.protocol["deadline_unix"]
        else:
            protocol.update(costs=cli.read(night.root / "configs/push_costs.json"),
                            time_bank_sha256=night.protocol["time_bank_sha256"],
                            runtime_limits={"deadline_unix": night.protocol["deadline_unix"]})
        atomic_json(root / "protocol.json", protocol)
        roots.append(root)
    return roots


@pytest.mark.parametrize("kind", ["rgb", "push"])
def test_planned_group_requires_complete_scientific_config(cli, night, kind):
    roots = _planned_group(cli, night, kind)
    args = SimpleNamespace(night_root=str(night.root))
    cli.require_planned_group(args, roots)
    path = roots[1] / "protocol.json"
    protocol = cli.read(path)
    field = "joint_pairs" if kind == "rgb" else "updates"
    protocol["config"][field] += 1
    atomic_json(path, protocol)
    with pytest.raises(ValueError, match="scientific config"):
        cli.require_planned_group(args, roots)


@pytest.mark.parametrize("field", ["costs", "time_bank_sha256"])
def test_planned_push_group_rejects_different_costs_or_bank(cli, night, field):
    roots = _planned_group(cli, night, "push")
    path = roots[1] / "protocol.json"
    protocol = cli.read(path)
    if field == "costs":
        protocol[field]["time"] += 1.
    else:
        protocol[field] = "different-bank"
    atomic_json(path, protocol)
    with pytest.raises(ValueError, match="costs/bank"):
        cli.require_planned_group(SimpleNamespace(night_root=str(night.root)), roots)


@pytest.mark.parametrize("kind", ["rgb", "push"])
@pytest.mark.parametrize("deadline", [None, float("nan"), "extension"])
def test_planned_group_rejects_missing_invalid_or_extended_deadlines(cli, night, kind, deadline):
    roots = _planned_group(cli, night, kind)
    path = roots[1] / "protocol.json"
    protocol = cli.read(path)
    scope = protocol if kind == "rgb" else protocol["runtime_limits"]
    scope["deadline_unix"] = night.protocol["deadline_unix"] + 1 if deadline == "extension" else deadline
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="deadline"):
        cli.require_planned_group(SimpleNamespace(night_root=str(night.root)), roots)


def test_planned_group_rejects_missing_duplicate_or_foreign_members(cli, night, tmp_path):
    roots = _planned_group(cli, night, "push")
    args = SimpleNamespace(night_root=str(night.root))
    for group in ([], [roots[0]], [roots[0], roots[0]]):
        with pytest.raises(ValueError, match="seed set"):
            cli.require_planned_group(args, group)
    with pytest.raises(ValueError, match="outside"):
        cli.require_planned_group(args, [roots[0], tmp_path / "foreign"])


@pytest.mark.parametrize("hours", [None, 0., -1., float("nan"), float("inf")])
def test_standalone_stage_requires_explicit_finite_budget(cli, hours):
    with pytest.raises(ValueError):
        cli.stage_guard(SimpleNamespace(night_root=None, max_hours=hours, output=None))


def _completed_push_fixture(root, seed):
    """Only freeze metadata is exercised; these bytes must never be torch-loaded."""
    root.mkdir()
    protocol = {"config": {"seed": seed, "updates": 2}, "time_bank_sha256": "shared-bank"}
    atomic_json(root / "protocol.json", protocol)
    names = ("initial.pt", "selected.pt", "final.pt")
    for name in names:
        (root / name).write_bytes((name + str(seed)).encode())
    atomic_json(root / "FROZEN.json", {
        "schema": "joint_push_v2", "selected_update": 1, "updates": 2,
        "protocol_sha256": file_sha256(root / "protocol.json"),
        "sha256": file_sha256(root / "selected.pt"),
        "checkpoint_sha256": {name: file_sha256(root / name) for name in names},
    })
    return root


def test_group_barrier_rejects_empty_duplicate_and_missing_seeds(tmp_path):
    from stabilization_learning.joint_v2.push import freeze_group
    a = _completed_push_fixture(tmp_path / "seed81", 81)
    b = _completed_push_fixture(tmp_path / "seed82", 82)
    target = tmp_path / "GROUP.json"
    for runs, seeds in (([], []), ([a, a], [81, 82]), ([a], [81, 82]), ([a, b], [81, 81])):
        with pytest.raises(ValueError):
            freeze_group(runs, target, seeds)
        assert not target.exists()
    result = freeze_group([a, b], target, [81, 82])
    assert result["expected_seeds"] == [81, 82] and len(result["runs"]) == 2
    with pytest.raises(FileExistsError):
        freeze_group([a, b], target, [81, 82])


def test_verify_is_explicit_and_uses_noninteractive_input(cli, monkeypatch):
    calls = []
    def record(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", record)
    assert cli.main(["verify"]) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:4] == [cli.sys.executable, "-m", "pytest", "-q"]
    assert kwargs["stdin"] == cli.subprocess.DEVNULL

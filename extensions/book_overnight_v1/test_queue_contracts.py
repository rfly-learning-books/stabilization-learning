"""CPU queue-control tests; no physical training or CUDA work."""
from datetime import datetime, timedelta, timezone
import sys

import pytest

from . import queue


def plan(tmp_path):
    rgb = tmp_path / "rgb"; rgb.mkdir()
    queue.write(rgb / "FINAL_STATE.json", dict(status="complete"))
    report = tmp_path / "rgb_report"; report.mkdir()
    queue.write(report / "REPORT.json", dict(root=str(rgb)))
    queue.write(report / "CURVES_COMPLETE.json", dict(complete=True))
    return dict(schema="book_overnight_queue_v1", deadline_iso=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),
        minimum_free_gib=0, checklist="E:/rfly-books/output/reports/全书实验验证任务清单_20261028/实验任务清单.json",
        after_rgb=dict(run=str(rgb), report=str(report), curves_marker=str(report / "CURVES_COMPLETE.json")),
        steps=[dict(id="cpu_fixture", task_ids=["E00"], scope="queue fixture, not scientific evidence", maximum_seconds=10,
                    commands=[dict(argv=[sys.executable, "-c", "print('fixture complete')"], cwd=str(tmp_path))], output_dirs=[], source_files=[])])


def test_queue_requires_rgb_delivery_and_preserves_task_matrix(tmp_path):
    cfg = plan(tmp_path)
    source = tmp_path / "plan.json"; queue.write(source, cfg)
    root = tmp_path / "queue"
    queue.prepare(source, root)
    result = queue.run(root)
    assert result["status"] == "complete"
    assert len(queue.read(root / "TASK_STATUS_FINAL.json")) == 32
    assert not (root / "WRITER.lock").exists()
    with pytest.raises(RuntimeError, match="not resumed"):
        queue.run(root)


def test_missing_rgb_curves_blocks_before_worker_lock(tmp_path):
    cfg = plan(tmp_path)
    cfg["after_rgb"]["curves_marker"] = str(tmp_path / "not_delivered")
    source = tmp_path / "plan.json"; queue.write(source, cfg)
    root = tmp_path / "queue"; queue.prepare(source, root)
    with pytest.raises(RuntimeError, match="curves"):
        queue.run(root)
    assert not (root / "WRITER.lock").exists()


def test_failed_step_keeps_failure_and_runs_independent_following(tmp_path):
    cfg = plan(tmp_path)
    cfg["steps"].insert(0, dict(id="failed_fixture", task_ids=["E00"], scope="intentional failure fixture", maximum_seconds=5,
        commands=[dict(argv=[sys.executable, "-c", "raise SystemExit(7)"], cwd=str(tmp_path))]))
    source = tmp_path / "plan.json"; queue.write(source, cfg)
    root = tmp_path / "queue"; queue.prepare(source, root)
    result = queue.run(root)
    assert result["status"] == "partial"
    assert [r["status"] for r in result["steps"]] == ["failed", "complete"]
    assert result["steps"][0]["commands"][0]["exit_code"] == 7

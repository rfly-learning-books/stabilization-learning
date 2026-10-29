"""Runtime locks bind visible GPU identities without touching hidden GPUs."""
import json
from types import SimpleNamespace

import pytest
import torch

from extensions.reproducible_training_v4 import runtime


@pytest.mark.parametrize("visibility", ["-1", ""])
def test_hidden_cpu_manifest_never_queries_cuda(monkeypatch, visibility):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden(*args, **kwargs):
        pytest.fail("hidden CPU run must not query CUDA/NVIDIA")
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(runtime, "_cuda_driver_devices", forbidden)
    monkeypatch.setattr(runtime, "_nvidia_devices", forbidden)
    monkeypatch.setattr(runtime, "_cuda_runtime_version", forbidden)
    assert runtime.manifest()["gpu"] == {"visibility": "hidden", "devices": []}


def test_hiding_previously_initialized_cuda_does_not_forge_cpu_runtime(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="already initialized"):
        runtime.gpu_identity()


def fixture_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    rows = [dict(uuid="GPU-second", name="GPU B", pci_bus_id="0000:02:00.0", compute_capability=[8, 9]),
            dict(uuid="GPU-first", name="GPU A", pci_bus_id="0000:01:00.0", compute_capability=[8, 6])]
    inventory = {row["uuid"]: dict(name=row["name"], pci_bus_id="0000" + row["pci_bus_id"],
                                   driver_version="610.60") for row in rows}
    props = [SimpleNamespace(name=row["name"], major=row["compute_capability"][0],
                             minor=row["compute_capability"][1], total_memory=12878086144) for row in rows]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props[index])
    monkeypatch.setattr(runtime, "_cuda_driver_devices", lambda: (rows, 13030))
    monkeypatch.setattr(runtime, "_nvidia_devices", lambda: inventory)
    monkeypatch.setattr(runtime, "_cuda_runtime_version", lambda: 12010)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 8902)
    return rows, inventory, props


def test_visible_order_matches_actual_driver_uuid_not_smi_order(monkeypatch):
    fixture_gpu(monkeypatch)
    identity = runtime.gpu_identity()
    assert [row["uuid"] for row in identity["devices"]] == ["GPU-second", "GPU-first"]
    assert [row["logical_index"] for row in identity["devices"]] == [0, 1]
    assert identity["cuda_runtime_version"] == 12010
    assert identity["cuda_driver_api_version"] == 13030
    assert identity["cudnn_version"] == 8902
    assert all(row["driver_version"] == "610.60" for row in identity["devices"])


@pytest.mark.parametrize("corrupt", ["uuid", "count", "name", "bus", "capability", "memory"])
def test_incomplete_or_conflicting_hardware_identity_rejected(monkeypatch, corrupt):
    rows, inventory, props = fixture_gpu(monkeypatch)
    if corrupt == "uuid":
        rows[0]["uuid"] = "GPU-unknown"
    elif corrupt == "count":
        rows.pop()
    elif corrupt == "name":
        props[0].name = "different"
    elif corrupt == "bus":
        rows[0]["pci_bus_id"] = "0000:09:00.0"
    elif corrupt == "capability":
        rows[0]["compute_capability"] = [9, 0]
    else:
        props[0].total_memory = 0
    with pytest.raises(RuntimeError, match="identit"):
        runtime.gpu_identity()


def test_changed_gpu_driver_rejected_by_existing_lock(tmp_path, monkeypatch):
    _, inventory, _ = fixture_gpu(monkeypatch)
    lock = tmp_path / "runtime.json"
    lock.write_text(json.dumps(runtime.manifest()), encoding="utf-8")
    inventory["GPU-second"]["driver_version"] = "999.00"
    with pytest.raises(RuntimeError, match="gpu"):
        runtime.check_lock(tmp_path, {"runtime_lock": "runtime.json"})


def test_explicit_device_unavailable_is_not_empty_success(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.gpu_identity()


def test_real_visible_gpu_identity_and_runtime():
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"} or not torch.cuda.is_available():
        pytest.skip("requires an explicitly visible GPU")
    result = runtime.gpu_identity()
    assert len(result["devices"]) == torch.cuda.device_count()
    assert result["cuda_runtime_version"] > 0
    assert result["cuda_driver_api_version"] > 0
    assert result["cudnn_version"] > 0
    assert all(row["uuid"].startswith("GPU-") for row in result["devices"])

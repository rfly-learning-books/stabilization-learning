"""Explicit deterministic runtime and portable tensor/content fingerprints."""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import csv
import ctypes
import subprocess
import sys
import uuid

import numpy as np
import torch


def configure(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def system_identity():
    # platform.platform() may obtain a different Windows build from cmd.exe
    # when CREATE_NO_WINDOW/redirected handles are used. The direct API is
    # identical in foreground and hidden workers on the same running system.
    if sys.platform == "win32":
        version = sys.getwindowsversion()
        return f"Windows-{version.major}.{version.minor}.{version.build}-{version.service_pack}"
    return platform.platform()


def _nvidia_devices():
    """Read immutable NVIDIA identities, never utilization or other processes."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pci.bus_id",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, timeout=15,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Cannot verify NVIDIA GPU/driver identities")
    devices = {}
    for fields in csv.reader(result.stdout.splitlines()):
        if len(fields) != 4 or any(not part.strip() for part in fields):
            raise RuntimeError("Incomplete NVIDIA identity record")
        identity, name, driver, bus = [part.strip() for part in fields]
        if identity in devices or not identity.startswith("GPU-") or driver == "[N/A]":
            raise RuntimeError("Ambiguous NVIDIA identity record")
        devices[identity] = dict(name=name, driver_version=driver, pci_bus_id=bus.lower())
    if not devices:
        raise RuntimeError("NVIDIA identity query returned no GPUs")
    return devices


def _cuda_runtime_version():
    """Query the CUDART already loaded by Torch, not an arbitrary CUDA install."""
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        major = int(torch.version.cuda.split(".")[0])
        names = [f"cudart64_{major}.dll"] if major >= 12 else ["cudart64_110.dll"]
        handles = [(name, kernel.GetModuleHandleW(name)) for name in names]
        loaded = [(name, handle) for name, handle in handles if handle]
        if len(loaded) != 1:
            raise RuntimeError("Cannot identify the CUDART loaded by Torch")
        name, handle = loaded[0]
        library = ctypes.WinDLL(name, handle=handle)
    elif sys.platform.startswith("linux"):
        from pathlib import Path
        paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if "/libcudart.so" in line}
        if len(paths) != 1:
            raise RuntimeError("Cannot uniquely identify the loaded CUDART")
        library = ctypes.CDLL(paths.pop())
    else:
        raise RuntimeError("CUDA runtime identity is unsupported on this platform")
    version = ctypes.c_int()
    function = library.cudaRuntimeGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    if function(ctypes.byref(version)) != 0 or version.value <= 0:
        raise RuntimeError("Cannot query the loaded CUDA runtime version")
    return version.value


def _cuda_driver_devices():
    """Driver ordinals share CUDA_VISIBLE_DEVICES with Torch's CUDA runtime."""
    library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    def invoke(name, argtypes, *args):
        function = getattr(library, name)
        function.argtypes, function.restype = argtypes, ctypes.c_int
        if function(*args) != 0:
            raise RuntimeError("Cannot verify CUDA driver identity: " + name)
    pointer = ctypes.POINTER(ctypes.c_int)
    invoke("cuInit", [ctypes.c_uint], 0)
    count, version = ctypes.c_int(), ctypes.c_int()
    invoke("cuDeviceGetCount", [pointer], ctypes.byref(count))
    invoke("cuDriverGetVersion", [pointer], ctypes.byref(version))
    devices = []
    uuid_function = "cuDeviceGetUuid_v2" if hasattr(library, "cuDeviceGetUuid_v2") else "cuDeviceGetUuid"
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        invoke("cuDeviceGet", [pointer, ctypes.c_int], ctypes.byref(device), index)
        identity, name, bus = ctypes.create_string_buffer(16), ctypes.create_string_buffer(256), ctypes.create_string_buffer(64)
        invoke(uuid_function, [ctypes.c_void_p, ctypes.c_int], identity, device)
        invoke("cuDeviceGetName", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], name, len(name), device)
        invoke("cuDeviceGetPCIBusId", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus, len(bus), device)
        invoke("cuDeviceComputeCapability", [pointer, pointer, ctypes.c_int], ctypes.byref(major), ctypes.byref(minor), device)
        devices.append(dict(uuid="GPU-" + str(uuid.UUID(bytes=identity.raw)), name=name.value.decode(),
                            pci_bus_id=bus.value.decode().lower(), compute_capability=[major.value, minor.value]))
    if version.value <= 0:
        raise RuntimeError("Invalid CUDA driver API version")
    return devices, version.value


def gpu_identity():
    """Bind actual visible devices and driver; hidden CPU runs never probe CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was already initialized before declaring a hidden CPU runtime")
        return dict(visibility="hidden", devices=[])
    if torch.version.cuda is None:
        return dict(visibility="unavailable", devices=[])
    count = torch.cuda.device_count()
    if count == 0:
        if visible is not None:
            raise RuntimeError("Declared CUDA devices are unavailable")
        return dict(visibility="unavailable", devices=[])
    # Initialize the same runtime the training backend uses before inspecting
    # its loaded library. No tensor allocation or RNG consumption is needed.
    properties = [torch.cuda.get_device_properties(index) for index in range(count)]
    driver_devices, driver_api = _cuda_driver_devices()
    inventory = _nvidia_devices()
    if len(driver_devices) != count or len({row["uuid"] for row in driver_devices}) != count:
        raise RuntimeError("Torch/driver visible GPU identities disagree")
    def bus_key(value):
        domain, bus, address = value.split(":")
        device, function = address.split(".")
        return tuple(int(part, 16) for part in (domain, bus, device, function))
    devices = []
    for index, (prop, identity) in enumerate(zip(properties, driver_devices)):
        nvidia = inventory.get(identity["uuid"])
        if (nvidia is None or identity["name"] != prop.name or nvidia["name"] != prop.name or
                identity["compute_capability"] != [prop.major, prop.minor] or
                bus_key(identity["pci_bus_id"]) != bus_key(nvidia["pci_bus_id"]) or prop.total_memory <= 0):
            raise RuntimeError("Cannot establish complete visible GPU identity")
        devices.append(dict(logical_index=index, name=prop.name, uuid=identity["uuid"],
                            pci_bus_id=nvidia["pci_bus_id"], compute_capability=[prop.major, prop.minor],
                            total_memory_bytes=prop.total_memory, driver_version=nvidia["driver_version"]))
    return dict(visibility="visible", devices=devices, cuda_driver_api_version=driver_api,
                cuda_runtime_version=_cuda_runtime_version(), cudnn_version=torch.backends.cudnn.version())


def manifest():
    packages = sorted([dist.metadata.get("Name", ""), dist.version]
                      for dist in importlib.metadata.distributions())
    return dict(schema="training_runtime_v4", python=sys.version, executable=sys.executable,
                platform=system_identity(), machine=platform.machine(), processor=platform.processor(),
                packages=packages, torch=torch.__version__, numpy=np.__version__,
                torch_cuda_build=torch.version.cuda, gpu=gpu_identity(), cpu_threads=torch.get_num_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                environment={k: os.environ.get(k) for k in ("PYTHONHASHSEED", "PYTHONUTF8",
                    "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                scope="Exact replay is checked within the recorded runtime/hardware; cross-device bitwise identity is not asserted")


def check_lock(project, config):
    """Reject a changed interpreter/dependency/hardware contract before training."""
    from pathlib import Path
    declared = config.get("runtime_lock")
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        raise ValueError("runtime_lock must name a JSON runtime manifest")
    path = Path(declared).expanduser()
    path = (path if path.is_absolute() else Path(project)/path).resolve()
    expected = json.loads(path.read_text(encoding="utf-8-sig"))
    actual = manifest()
    if expected != actual:
        changed = sorted(k for k in set(expected)|set(actual) if expected.get(k) != actual.get(k))
        detail = {key: dict(expected=expected.get(key), actual=actual.get(key)) for key in changed if key != "packages"}
        raise RuntimeError("Runtime differs from frozen environment lock: " + ", ".join(changed) + "; " + repr(detail))
    return path


def assert_contract():
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("A backend changed the declared deterministic/single-thread runtime")
    if torch.backends.cudnn.benchmark or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("A backend changed the declared numerical precision/runtime policy")


def content_hash(value):
    """Hash tensor contents, not torch.save zip metadata or output file names."""
    digest = hashlib.sha256()
    def frame(tag, payload):
        digest.update(tag + str(len(payload)).encode() + b":" + payload + b";")
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            frame(b"tensor_meta", (str(tensor.dtype)+repr(tuple(tensor.shape))).encode())
            frame(b"tensor_data", tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            frame(b"array_meta", (str(item.dtype)+repr(item.shape)).encode())
            frame(b"array_data", np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict" + str(len(item)).encode() + b"{")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(item[key])
            digest.update(b"}end_dict")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode() + b"[")
            for part in item:
                visit(part)
            digest.update(b"]end_sequence")
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, (str, int, float, bool)) or item is None:
            frame(type(item).__name__.encode(), repr(item).encode())
        else:
            raise TypeError("Unfingerprinted checkpoint type: " + str(type(item)))
    visit(value)
    return digest.hexdigest()


# These fields measure execution or publication, never learned/scientific values.
VOLATILE = {"wall_seconds", "wall_time", "elapsed_seconds", "started_unix", "finished_unix",
            "created_unix", "created_utc", "frozen_unix", "deadline_unix", "wall_time_seconds",
            "group_freeze_sha256", "protocol_sha256", "suite_protocol_sha256", "sha256",
            "path", "checkpoint", "source_digest"}


def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, (tuple, list)):
        return [scientific(v) for v in value]
    return value


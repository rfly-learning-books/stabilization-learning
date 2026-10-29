"""Check the actual Git publication set for data, weights and credentials."""
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {"runs", "artifacts", "handoff", "runtime_logs", "replay", "checkpoints", "external_assets", "__pycache__", "build", "dist"}
BLOCKED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npz", ".npy", ".pkl", ".pickle", ".h5", ".hdf5", ".bin", ".joblib", ".parquet", ".csv", ".jsonl", ".log", ".gz", ".zip", ".mp4", ".avi", ".webm", ".gif", ".pem", ".key"}
SECRET_PATTERNS = [r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    r"\bAKIA[A-Z0-9]{16}\b", r"\bsk-proj-[A-Za-z0-9_-]{32,}\b"]

def audit():
    result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT, check=True, capture_output=True)
    names = sorted(set(result.stdout.decode("utf-8").strip("\0").split("\0")))
    failures, total = [], 0
    for name in names:
        p = ROOT / name
        if not p.is_file():
            failures.append({"file": name, "reason": "missing publication file"}); continue
        total += p.stat().st_size
        parts = Path(name).parts
        if set(parts) & BLOCKED_DIRS or any(x.startswith('.venv') for x in parts) or p.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append({"file": name, "reason": "data/weights/runtime artifact"})
        if p.stat().st_size > 20 * 1024 * 1024:
            failures.append({"file": name, "reason": "file exceeds 20 MiB review boundary"})
        if p.name in {".env", ".netrc", ".pypirc", "assets.json"} or p.name.startswith("runtime_") and "lock" in p.name:
            failures.append({"file": name, "reason": "local configuration"})
        if p.suffix in {".py", ".json", ".md", ".toml", ".txt", ".yml", ".yaml", ".ps1"}:
            value = p.read_text(encoding="utf-8-sig", errors="replace")
            if any(re.search(pattern, value) for pattern in SECRET_PATTERNS):
                failures.append({"file": name, "reason": "credential pattern; contents withheld"})
    return {"passed": not failures, "files": len(names), "bytes": total, "failures": failures}

if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)

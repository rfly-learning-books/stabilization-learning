"""Isolate strided RGB archive writes; no simulator, learner or old replay."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = np.random.default_rng(81).integers(0, 256, (64, 4, 256, 256, 3), dtype=np.uint8)
    view = raw[:, 0]
    assert not view.flags.c_contiguous
    rows = []
    hashes = set()
    for repeat in range(3):
        for mode in ("strided", "contiguous_copy_included"):
            path = args.output/f"{mode}_{repeat}.npy"
            start = time.perf_counter()
            array = view if mode == "strided" else np.ascontiguousarray(view)
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush(); os.fsync(stream.fileno())
            seconds = time.perf_counter()-start
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert np.array_equal(np.load(path, allow_pickle=False), view)
            hashes.add(digest)
            rows.append(dict(mode=mode, repeat=repeat, seconds=seconds, bytes=view.nbytes, sha256=digest))
    assert len(hashes) == 1, "The execution improvement must preserve all stored bytes"
    result = dict(rows=rows, identical_npy_bytes=True,
                  mean_seconds={mode: float(np.mean([r["seconds"] for r in rows if r["mode"] == mode]))
                                for mode in ("strided", "contiguous_copy_included")})
    (args.output/"RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["mean_seconds"]))


if __name__ == "__main__":
    main()

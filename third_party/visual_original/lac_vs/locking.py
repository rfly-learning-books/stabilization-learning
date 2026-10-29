"""OS-backed single-writer guard; process death releases the lock automatically."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def writer_lock(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep outside the run so prepare can still reject nonempty foreign runs.
    path = output.parent / f".{output.name}.writer.lock"
    stream = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(f"Another writer already owns run {output}") from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
    # Do not unlink: replacing a live lock path would permit two lock inodes.

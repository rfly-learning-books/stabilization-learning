"""Stop only the live subprocess tree owned by this supervisor.

Windows venv python.exe is a launcher with a second Python child. Terminating
only its Popen handle can leave the actual CUDA writer alive.
"""
import os
import signal
import subprocess


def terminate_owned_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdin=subprocess.DEVNULL, capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW, timeout=20)
        if result.returncode and process.poll() is None:
            raise RuntimeError("Unable to stop owned writer tree: " + result.stderr.decode(errors="replace"))
    else:
        # Caller creates a new session; no unrelated Python belongs to it.
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            raise RuntimeError("Owned writer tree did not exit after taskkill")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

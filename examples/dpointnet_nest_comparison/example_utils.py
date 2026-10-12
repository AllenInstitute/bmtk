import hashlib
import json
from pathlib import Path


def input_hashes():
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("network", "inputs", "components")
        for path in sorted(Path(folder).rglob("*"))
        if path.is_file()
    }


def validate_grid(dt, duration):
    build = json.loads(Path("network/build.json").read_text())
    if dt != 1.0:
        raise ValueError("This spike-file replay example requires dt=1 ms; see README.")
    if build["dt_ms"] != dt or build["duration_ms"] != duration:
        raise ValueError("Rebuild inputs for the configured timestep and duration.")

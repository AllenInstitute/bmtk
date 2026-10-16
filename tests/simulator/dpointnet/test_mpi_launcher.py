import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("failure,expected", [
    ("raise RuntimeError('intentional-worker-error')",
     "RuntimeError: intentional-worker-error"),
    ("raise SystemExit(3)", "SystemExit: 3"),
])
def test_launcher_flushes_buffered_output_before_abort(tmp_path, failure, expected):
    (tmp_path / "mpi4py.py").write_text(
        "import os\n"
        "class Communicator:\n"
        "    def allgather(self, value):\n"
        "        return [value]\n"
        "    def Abort(self, code):\n"
        "        os._exit(code)\n"
        "class MPI:\n"
        "    COMM_WORLD = Communicator()\n"
    )
    script = tmp_path / "worker.py"
    script.write_text("print('intentional-worker-stdout')\n" + failure + "\n")
    launcher = (
        Path(__file__).resolve().parents[3] / "bmtk/utils/dpointnet_mpi.py"
    )
    bootstrap = (
        "import io, runpy, sys\n"
        "original_stdout, original_stderr = sys.stdout, sys.stderr\n"
        "sys.stdout = io.TextIOWrapper(original_stdout.buffer, "
        "line_buffering=False, write_through=False)\n"
        "sys.stderr = io.TextIOWrapper(original_stderr.buffer, "
        "line_buffering=False, write_through=False)\n"
        "launcher = sys.argv[1]\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(launcher, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(launcher), "--cpu", str(script)],
        env=dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert expected in result.stderr, result.stdout + result.stderr
    assert "intentional-worker-stdout" in result.stdout

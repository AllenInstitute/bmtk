import csv
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import numpy as np

pytest.importorskip("tensorflow")

import h5py

from bmtk.simulator.dpointnet.config import Config
from bmtk.simulator.dpointnet.io_tools import RNNIOUtils
from bmtk.utils.io.ioutils import bmtk_world_comm


@pytest.fixture
def local_io(monkeypatch):
    monkeypatch.setattr(RNNIOUtils, "_logger", logging.Logger("mpi-test"))
    monkeypatch.delenv("BMTK_DPOINTNET_MPI_LAUNCH", raising=False)
    value = RNNIOUtils()
    yield value
    for handler in value.logger.handlers:
        handler.close()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("overwrite", [False, True])
def test_mpi_output_and_config_have_one_owner(
    local_io, monkeypatch, tmp_path, rank, overwrite,
):
    barriers = []
    communicator = SimpleNamespace(
        Get_rank=lambda: rank, Get_size=lambda: 2,
        Barrier=lambda: barriers.append(True),
    )
    monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "cpu")
    monkeypatch.setattr(bmtk_world_comm, "_comm", communicator)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.touch()
    config = Config.from_dict({
        "config_path": str(tmp_path / "input.json"),
        "run": {"distribution": {"strategy": "mpi"}},
        "output": {
            "output_dir": str(output), "log_file": "run.log",
            "overwrite_output_dir": overwrite,
        },
    })
    config.io = local_io
    config.build_env()
    assert (local_io.mpi_rank, local_io.mpi_size) == (rank, 2)
    assert len(barriers) == 3
    assert sentinel.exists() == (rank != 0 or not overwrite)
    assert (output / "input.json").exists() == (rank == 0)
    handlers = [
        handler for handler in local_io.logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(handlers) == int(rank == 0)
    local_io.log_info("chief-log-token")
    if rank == 0:
        assert (output / "run.log").read_text().count("chief-log-token") == 1
    sentinel.touch()
    config.build_env()
    assert sentinel.exists()
    assert len(barriers) == 3
    config.build_env(force=True)
    assert len(barriers) == 6
    assert sentinel.exists() == (rank != 0 or not overwrite)
    assert len([
        handler for handler in local_io.logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]) == int(rank == 0)


def test_serial_io_does_not_require_mpi(local_io, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setattr(bmtk_world_comm, "_comm", object())
    local_io.configure_mpi()
    assert (local_io.mpi_rank, local_io.mpi_size) == (0, 1)
    output = tmp_path / "serial"
    local_io.setup_output_dir(str(output), str(output / "run.log"))
    assert (output / "run.log").is_file()


def test_mpi_io_rejects_unavailable_communicator(local_io, monkeypatch):
    monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "cpu")
    monkeypatch.setattr(type(bmtk_world_comm), "comm", property(lambda self: None))
    with pytest.raises(RuntimeError, match="communicator"):
        local_io.configure_mpi()


def test_mpi_io_rejects_unknown_launch_mode(local_io, monkeypatch):
    monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "unknown")
    with pytest.raises(ValueError, match="launcher mode"):
        local_io.configure_mpi()


def test_rank_local_fatal_log_does_not_wait_for_peers(local_io, monkeypatch):
    barriers = []
    monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "cpu")
    monkeypatch.setattr(bmtk_world_comm, "_comm", SimpleNamespace(
        Get_rank=lambda: 1, Get_size=lambda: 2,
        Barrier=lambda: barriers.append(True),
    ))
    local_io.configure_mpi()
    with pytest.raises(Exception, match="rank-local failure"):
        local_io.log_exception("rank-local failure")
    assert barriers == []


@pytest.mark.parametrize("launched", [False, True])
def test_incorrect_launcher_config_pair_fails_before_output(
    local_io, monkeypatch, tmp_path, launched,
):
    if launched:
        monkeypatch.setenv("BMTK_DPOINTNET_MPI_LAUNCH", "cpu")
    config = Config.from_dict({
        "run": {} if launched else {"distribution": {"strategy": "mpi"}},
        "output": {"output_dir": str(tmp_path / "must-not-exist")},
    })
    config.io = local_io
    with pytest.raises(ValueError, match="MPI"):
        config.build_env()
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.skipif(
    os.environ.get("BMTK_RUN_MPI_TESTS") != "1",
    reason="Real MPI subprocess tests require explicit BMTK_RUN_MPI_TESTS=1.",
)
@pytest.mark.parametrize("mode", [
    "serial", "one-rank", "two-rank", "configured-comm",
    "setup-error", "copy-error", "log-error",
])
def test_real_config_lifecycle(mode, tmp_path):
    launcher = shutil.which("mpiexec")
    timeout = shutil.which("timeout")
    assert launcher and timeout, "MPI integration requires mpiexec and GNU timeout."
    probe = Path(__file__).with_name("config_parallel_probe.py")
    command = [timeout, "--kill-after=10s", "90s"]
    if mode == "serial":
        command += [sys.executable, str(probe), str(tmp_path), "--serial"]
    else:
        command += [
            launcher, "-n", "1" if mode == "one-rank" else "2",
            sys.executable, "-m", "bmtk.utils.dpointnet_mpi", "--cpu",
            str(probe), str(tmp_path), "--mode", mode,
        ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120,
        env=dict(
            os.environ, CUDA_VISIBLE_DEVICES="", TF_NUM_INTRAOP_THREADS="1",
            TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", BMTK_DPOINTNET_DISABLE_FUSED_CUDA="1",
        ),
    )
    output = result.stdout + result.stderr
    assert result.returncode not in (124, 137), output
    if mode in ("setup-error", "copy-error", "log-error"):
        assert result.returncode != 0, output
        assert f"intentional-{mode}" in output
    else:
        assert result.returncode == 0, output
        report = json.loads((tmp_path / "result.json").read_text())
        assert report["status"] == "passed"
        assert len(report["ranks"]) == (1 if mode in ("serial", "one-rank") else 2)


@pytest.mark.skipif(
    os.environ.get("BMTK_RUN_MPI_TESTS") != "1",
    reason="Real MPI subprocess tests require explicit BMTK_RUN_MPI_TESTS=1.",
)
@pytest.mark.parametrize("serial", [False, True])
def test_real_documented_training_example(serial, tmp_path):
    launcher = shutil.which("mpiexec")
    timeout = shutil.which("timeout")
    assert launcher and timeout, "MPI integration requires mpiexec and GNU timeout."
    example = Path(__file__).resolve().parents[3] / "examples/dpointnet_mpi/run_training.py"
    output = tmp_path / "example"
    command = [timeout, "--kill-after=10s", "180s"]
    if not serial:
        command += [launcher, "-n", "2"]
    command += [sys.executable]
    if not serial:
        command += ["-m", "bmtk.utils.dpointnet_mpi", "--cpu"]
    command += [str(example), "--output", str(output)]
    if serial:
        command += ["--serial"]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=210,
        env=dict(
            os.environ, CUDA_VISIBLE_DEVICES="", TF_NUM_INTRAOP_THREADS="1",
            TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", BMTK_DPOINTNET_DISABLE_FUSED_CUDA="1",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = (output / "run.log").read_text()
    assert log.count("Completed 4 optimizer updates.") == 1
    config = json.loads((output / "config.train.json").read_text())
    assert ("distribution" not in config["run"]) == serial
    with (output / "callbacks/losses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    steps = [
        row for row in rows
        if row["loss_type"] == "step" and row["loss_function"] == "__total_loss"
    ]
    assert len(steps) == 4
    assert {(row["epoch"], row["step"]) for row in steps} == {
        ("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"),
    }
    assert all(np.isfinite(float(row["loss_value"])) for row in rows)
    for filename in ("glifs_glifs_edges.h5", "virts_glifs_edges.h5"):
        path = output / "callbacks/trained_weights" / filename
        with h5py.File(path) as stream:
            weights = []
            stream.visititems(
                lambda name, value: weights.append(value[:])
                if name.endswith("/syn_weight") else None
            )
        assert weights and all(
            np.isfinite(value).all() and value.size > 0 for value in weights
        )

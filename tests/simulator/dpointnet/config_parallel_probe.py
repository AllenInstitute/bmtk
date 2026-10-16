"""Real-process oracle for normal Config.build_env ownership and failure cleanup."""

import argparse
import json
import logging
from pathlib import Path

from mpi4py import MPI

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet.io_tools import io
from bmtk.utils.io.ioutils import set_world_comm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--mode", default="two-rank")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    if args.mode == "configured-comm":
        comm = comm.Dup()
        set_world_comm(comm)
    directory = args.directory
    path = directory / "config.json"
    output = directory / "output"
    if comm.rank == 0:
        output.mkdir()
        (output / "stale").touch()
        path.write_text(json.dumps({
            "run": {
                "seq_len": 9, "batch_size": 2, "dtype": "float32",
                **({} if args.serial else {"distribution": {"strategy": "mpi"}}),
            },
            "output": {
                "output_dir": str(output), "log_file": "run.log",
                "overwrite_output_dir": True,
            },
        }))
    comm.Barrier()
    config = dpointnet.Config.from_json(str(path))
    copies = []
    original_copy = config.copy_to_output

    def copy_config():
        if comm.rank == 0:
            copies.append(True)
            if args.mode == "copy-error":
                raise OSError("intentional-copy-error")
        original_copy()

    config.copy_to_output = copy_config
    if args.mode == "log-error" and comm.rank == 1:
        io.log_exception("intentional-log-error")
    if args.mode == "setup-error" and comm.rank == 0:
        def fail_setup(*args, **kwargs):
            raise OSError("intentional-setup-error")
        io.setup_output_dir = fail_setup
    config.build_env()
    assert (io.mpi_rank, io.mpi_size) == (comm.rank, comm.size)
    assert not (output / "stale").exists()
    assert json.loads((output / path.name).read_text())["run"]["batch_size"] == 2
    if not args.serial:
        from bmtk.simulator.dpointnet.process_distribution import ProcessDataParallel

        parallel = ProcessDataParallel({"strategy": "mpi"}, 2, "float32")
        assert parallel.comm is comm
        assert parallel.replicas == comm.size
    io.log_info("chief-only-token")
    for handler in io.logger.handlers:
        handler.flush()
    comm.Barrier()
    assert (output / "run.log").read_text().count("chief-only-token") == 1
    if comm.rank == 0:
        (output / "keep").touch()
    comm.Barrier()
    config.build_env()
    assert (output / "keep").exists()
    config.build_env(force=True)
    assert not (output / "keep").exists()
    assert len(copies) == 2 * int(comm.rank == 0)
    handlers = [
        handler for handler in io.logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    assert len(handlers) == int(comm.rank == 0)
    ranks = comm.gather({
        "rank": comm.rank, "size": comm.size, "config_copies": len(copies),
        "file_handlers": len(handlers),
    }, root=0)
    if comm.rank == 0:
        (directory / "result.json").write_text(json.dumps({
            "status": "passed", "mode": args.mode, "ranks": ranks,
        }, indent=2) + "\n")
    comm.Barrier()
    if args.mode == "configured-comm":
        set_world_comm(MPI.COMM_WORLD)
        comm.Free()


if __name__ == "__main__":
    main()

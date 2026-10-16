"""Launch one DPointNet process per allocated GPU before TensorFlow is imported."""

import argparse
import os
from pathlib import Path
import runpy
import socket
import sys
import traceback


def select_device(allocation, rank, size):
    if allocation is None:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the allocated GPU list.")
    devices = [value.strip() for value in allocation.split(",") if value.strip()]
    if "-1" in devices or len(devices) != size or len(set(devices)) != size:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly one distinct GPU per MPI rank."
        )
    return devices[rank]


def _abort_with_traceback(comm):
    try:
        traceback.print_exc()
        # MPI abort bypasses Python shutdown and its buffered stream flushing.
        sys.stderr.flush()
        sys.stdout.flush()
    finally:
        comm.Abort(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU qualification.")
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    try:
        if len(set(comm.allgather(socket.gethostname()))) != 1:
            raise ValueError("DPointNet MPI training currently supports one host only.")
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            "" if args.cpu else select_device(
                os.environ.get("CUDA_VISIBLE_DEVICES"), comm.rank, comm.size
            )
        )
        os.environ["BMTK_DPOINTNET_MPI_LAUNCH"] = "cpu" if args.cpu else "gpu"
        os.environ.pop("TF_GPU_ALLOCATOR", None)
        if "tensorflow" in sys.modules:
            raise RuntimeError("GPU selection must precede TensorFlow import.")
        script = args.script.resolve(strict=True)
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *args.arguments]
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            _abort_with_traceback(comm)
    except BaseException:
        _abort_with_traceback(comm)


if __name__ == "__main__":
    main()

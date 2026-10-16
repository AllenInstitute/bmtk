"""Run the small SONATA training example through the normal BMTK config API."""

import argparse
import os
from pathlib import Path

os.environ.pop("TF_GPU_ALLOCATOR", None)

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet.io_tools import io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", type=Path,
        default=Path(__file__).with_name("config.train.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("output_mpi"))
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    config = dpointnet.Config.from_json(
        str(args.config), output_dir=str(args.output.resolve())
    )
    if args.serial:
        config.run.pop("distribution", None)
    config.build_env()
    network = dpointnet.RNN.from_config(config)
    network.run()
    io.log_info(
        f"Completed {int(network.training_engine.optimizer.iterations.numpy())} "
        "optimizer updates."
    )


if __name__ == "__main__":
    main()

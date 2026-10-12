import argparse
import json
from pathlib import Path

import nest

from bmtk.simulator import pointnet
from example_utils import input_hashes, validate_grid


def run(config_path):
    config = pointnet.Config.from_json(config_path)
    validate_grid(config["run"]["dt"], config["run"]["tstop"])
    config.build_env()
    network = pointnet.PointNetwork.from_config(config)
    simulation = pointnet.PointSimulator.from_config(config, network)
    simulation.run()
    output = Path(config["output"]["output_dir"])
    (output / "resolved.json").write_text(
        json.dumps(
            {
                "nest": nest.__version__,
                "bmtk_source": pointnet.__file__,
                "dt_ms": config["run"]["dt"],
                "reset": "NEST glif_psc hard reset",
                "precision": "NEST internal double precision, not FP32",
                "input_sha256": input_hashes(),
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the actual NEST reference through PointNet.")
    parser.add_argument("config", nargs="?", default="config.nest.json")
    run(parser.parse_args().config)

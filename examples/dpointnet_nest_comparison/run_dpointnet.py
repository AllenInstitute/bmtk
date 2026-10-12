import argparse
import json
import os
from pathlib import Path

os.environ.pop("TF_GPU_ALLOCATOR", None)

import tensorflow as tf
import numpy as np

from bmtk.simulator import dpointnet
from example_utils import input_hashes, validate_grid


def merge(base, overrides):
    for name, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(name), dict):
            merge(base[name], value)
        else:
            base[name] = value
    return base


def run(config_path):
    base = json.loads(Path("config.base.json").read_text())
    profile = json.loads(Path(config_path).read_text())
    complete = merge(base, profile)
    validate_grid(complete["run"]["dt"], complete["run"]["dt"] * complete["run"]["seq_len"])
    config = dpointnet.Config.from_dict(complete)
    config.build_env()
    network = dpointnet.RNN.from_config(config)
    try:
        network.run()
        cell = network.cell
        report = {
            "config": str(config_path),
            "tensorflow": tf.__version__,
            "bmtk_source": dpointnet.__file__,
            "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "dt_ms": float(network.dt),
            "dynamics_mode": cell.dynamics_mode,
            "hard_reset": profile["rnn_cell_params"]["hard_reset"],
            "hard_reset_gradient_mode": cell.hard_reset_gradient_mode,
            "compute_dtype": str(cell.compute_dtype),
            "state_precision": cell.state_precision,
            "voltage_and_asc_dtype": cell.state_dtype.name,
            "acceleration": cell.acceleration_report,
            "input_sha256": input_hashes(),
        }
        if network.training_engine is not None:
            variables = network.model.trainable_variables
            if not all(np.isfinite(variable.numpy()).all() for variable in variables):
                raise ValueError("Training produced non-finite model variables.")
            report["optimizer_updates"] = int(
                network.training_engine.optimizer.iterations.numpy()
            )
        output = Path(config["output"]["output_dir"])
        (output / "resolved.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        network.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one reset/precision comparison profile.")
    parser.add_argument("config", nargs="?", default="config.fp32.hard.json")
    run(parser.parse_args().config)

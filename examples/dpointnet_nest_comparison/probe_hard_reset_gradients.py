import json
import os
from pathlib import Path

os.environ.pop("TF_GPU_ALLOCATOR", None)

import h5py
import numpy as np
import tensorflow as tf

from bmtk.simulator import dpointnet
from plot_output import read_spikes, read_voltages
from run_dpointnet import merge


def run():
    config_dict = json.loads(Path("config.base.json").read_text())
    merge(config_dict, json.loads(Path("config.train.hard_surrogate.json").read_text()))
    config_dict.pop("training")
    config_dict["output"]["output_dir"] = "output_gradient_probe"
    config = dpointnet.Config.from_dict(config_dict)
    config.build_env()
    network = dpointnet.RNN.from_config(config)
    try:
        network.build(training=False)
        sequence = np.zeros((1, network.seq_len, 100), dtype=np.float32)
        with h5py.File("inputs/spikes.h5") as handle:
            group = handle["spikes/virts"]
            times, ids = group["timestamps"][:], group["node_ids"][:]
        steps = np.floor(times / network.dt).astype(int)
        sequence[0, steps, ids] = 1.0
        cell = network.cell
        state = list(cell.zero_state(1, cell.compute_dtype))
        state[1] = tf.ones_like(state[1]) * 1.5
        measurements = {}
        forward_results = {}
        baseline_spikes = read_spikes(Path("output_fp32_hard"), network.dt, network.seq_len, 300)
        baseline_voltages = read_voltages(Path("output_fp32_hard"), network.dt, network.seq_len, 300)
        for mode in ("exact", "soft_surrogate"):
            cell.hard_reset_gradient_mode = mode
            with tf.GradientTape() as tape:
                tape.watch(state[1])
                output, after_first = cell(tf.zeros((1, 100), tf.float32), state)
                output, after_second = cell(tf.zeros((1, 100), tf.float32), after_first)
                voltage_loss = tf.reduce_sum(after_second[1])
            dv = tape.gradient(voltage_loss, state[1])
            measurements[mode] = {
                "initial_voltage_gradient_l2": float(tf.linalg.norm(dv)),
                "first_step_spikes": int(tf.reduce_sum(after_first[0][:, :300])),
            }
            forward_results[mode] = [value.numpy() for value in after_second]
        for exact, surrogate in zip(forward_results["exact"], forward_results["soft_surrogate"]):
            np.testing.assert_array_equal(exact, surrogate)
        if measurements["exact"]["initial_voltage_gradient_l2"] != 0:
            raise AssertionError("Exact hard reset unexpectedly retained voltage credit.")
        if measurements["soft_surrogate"]["initial_voltage_gradient_l2"] <= 0:
            raise AssertionError("Surrogate voltage credit was not restored.")

        # Each mode gets its own graph so tracing cannot retain the previous rule.
        for mode in ("exact", "soft_surrogate"):
            network.cell_params["hard_reset_gradient_mode"] = mode
            network.build(rebuild=True, training=False)
            cell = network.cell
            with tf.GradientTape() as tape:
                outputs = network.run_extractor(
                    tf.constant(sequence), cell.zero_state(1, cell.compute_dtype)
                )
                spikes = tf.cast(outputs[0][0], tf.float32)
                rates = tf.reduce_sum(spikes, axis=1) * (
                    1000.0 / (network.seq_len * network.dt)
                )
                loss = tf.reduce_mean(tf.square(rates - 30.0))
            variables = network.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            if any(gradient is None for gradient in gradients):
                raise ValueError("A trainable weight group has disconnected gradients.")
            if not all(np.isfinite(gradient.numpy()).all() for gradient in gradients):
                raise ValueError("Weight gradients are non-finite.")
            measurements[mode].update(
                loss=float(loss),
                spike_count=int(tf.reduce_sum(spikes)),
                weight_gradient_l2={
                    variable.name: float(tf.linalg.norm(gradient))
                    for variable, gradient in zip(variables, gradients)
                },
            )
            np.testing.assert_array_equal(spikes.numpy()[0], baseline_spikes)
            np.testing.assert_array_equal(outputs[0][1].numpy()[0], baseline_voltages)
            forward_results[mode] = [value.numpy() for value in tf.nest.flatten(outputs)]
        for exact, surrogate in zip(forward_results["exact"], forward_results["soft_surrogate"]):
            np.testing.assert_array_equal(exact, surrogate)
        report = {
            "hard_forward_identical": True,
            "baseline_spikes_and_voltages_identical": True,
            "reset_and_refractory_voltage_credit_restored": True,
            "measurements": measurements,
        }
        output = Path("output_gradient_probe")
        (output / "gradient_probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        network.cleanup()


if __name__ == "__main__":
    run()

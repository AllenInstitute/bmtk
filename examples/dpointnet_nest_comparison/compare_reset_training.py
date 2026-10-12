import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.pop("TF_GPU_ALLOCATOR", None)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from bmtk.simulator import dpointnet
from example_utils import input_hashes
from plot_output import difference, read_spikes, read_voltages
from run_dpointnet import merge


ARMS = ("hard_surrogate", "soft")
LABELS = {"hard_surrogate": "Hard + surrogate", "soft": "Soft reset"}


def complete_config(arm):
    return merge(
        json.loads(Path("config.base.json").read_text()),
        json.loads(Path(f"config.train.{arm}.json").read_text()),
    )


def history(arm):
    with Path(f"output_train_{arm}/callbacks/losses.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["loss_function"] == "__total_loss"]
    steps = [float(row["loss_value"]) for row in rows if row["loss_type"] == "step"]
    validation = [float(row["loss_value"]) for row in rows if row["loss_type"] == "validation"]
    if len(steps) != 6 or len(validation) != 2 or not np.isfinite(steps + validation).all():
        raise ValueError(f"Incomplete or non-finite six-update history for {arm}.")
    return steps, validation


def initial_gradients(arm):
    config = complete_config(arm)
    config.pop("training")
    config["output"]["output_dir"] = f"output_probe_{arm}"
    config = dpointnet.Config.from_dict(config)
    config.build_env()
    network = dpointnet.RNN.from_config(config)
    try:
        network.build(training=False)
        cell = network.cell
        state = list(cell.zero_state(1, cell.compute_dtype))
        state[1] = tf.ones_like(state[1]) * 1.5
        with tf.GradientTape() as tape:
            tape.watch(state[1])
            _, first = cell(tf.zeros((1, 100), tf.float32), state)
            _, second = cell(tf.zeros((1, 100), tf.float32), first)
            voltage_loss = tf.reduce_sum(second[1])
        voltage_gradient = tape.gradient(voltage_loss, state[1])
        if int(tf.reduce_sum(first[0][:, :300])) != 300:
            raise ValueError("The forced-reset probe did not spike every neuron.")
        sequence = np.zeros((1, 300, 100), dtype=np.float32)
        with h5py.File("inputs/spikes.h5") as handle:
            group = handle["spikes/virts"]
            sequence[0, np.floor(group["timestamps"][:] / network.dt).astype(int), group["node_ids"][:]] = 1
        with tf.GradientTape() as tape:
            outputs = network.run_extractor(tf.constant(sequence), cell.zero_state(1, cell.compute_dtype))
            spikes = tf.cast(outputs[0][0], tf.float32)
            loss = tf.reduce_mean(tf.square(tf.reduce_sum(spikes, axis=1) / 0.3 - 30.0))
        variables = network.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if voltage_gradient is None or any(value is None for value in gradients):
            raise ValueError("Disconnected voltage or trainable-weight gradient.")
        values = [voltage_gradient.numpy(), *(value.numpy() for value in gradients)]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("Non-finite initial gradients.")
        return {
            "forced_reset_voltage_gradient_l2": float(tf.linalg.norm(voltage_gradient)),
            "initial_rate_mse": float(loss),
            "initial_spikes": int(tf.reduce_sum(spikes)),
            "weight_gradients": {variable.name: value.numpy() for variable, value in zip(variables, gradients)},
        }
    finally:
        network.cleanup()


def compare():
    configs = {arm: complete_config(arm) for arm in ARMS}
    normalized = copy.deepcopy(configs)
    for config in normalized.values():
        config["rnn_cell_params"].pop("hard_reset")
        config["rnn_cell_params"].pop("hard_reset_gradient_mode")
        config["training"]["callbacks"].pop("callbacks_dir")
        config["output"].pop("output_dir")
    if normalized[ARMS[0]] != normalized[ARMS[1]]:
        raise ValueError("The training recipes differ beyond reset policy and output paths.")
    hashes = input_hashes()
    histories = {arm: history(arm) for arm in ARMS}
    probes = {}
    # SONATA population ID maps are process-global; isolate each model load.
    for arm in ARMS:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe", arm], check=True)
        folder = Path(f"output_probe_{arm}")
        probes[arm] = json.loads((folder / "initial_gradients.json").read_text())
        with np.load(folder / "weight_gradients.npz", allow_pickle=False) as gradients:
            probes[arm]["weight_gradients"] = {name: gradients[name] for name in gradients.files}
    gradient_comparison = {}
    for name in probes[ARMS[0]]["weight_gradients"]:
        hard, soft = (probes[arm]["weight_gradients"][name].astype(np.float64) for arm in ARMS)
        norms = [float(np.linalg.norm(value)) for value in (hard, soft)]
        if min(norms) == 0:
            raise ValueError(f"Expected nonzero initial weight gradients for {name}.")
        gradient_comparison[name] = {
            "hard_surrogate_l2": norms[0], "soft_l2": norms[1],
            "cosine_similarity": float(np.vdot(hard, soft) / np.prod(norms)),
        }
    for probe in probes.values():
        probe.pop("weight_gradients")
    evaluations = {}
    events = {}
    for arm in ARMS:
        resolved = json.loads(Path(f"output_train_{arm}/resolved.json").read_text())
        if resolved["gpus"] or resolved["optimizer_updates"] != 6 or resolved["compute_dtype"] != "float32":
            raise ValueError(f"Unexpected training execution for {arm}.")
        if resolved["input_sha256"] != hashes:
            raise ValueError("Training inputs or network assets changed.")
        np.testing.assert_allclose(probes[arm]["initial_rate_mse"], histories[arm][0][0], rtol=1e-6, atol=1e-6)
        evaluations[arm] = {}
        for reset in ("hard", "soft"):
            config = copy.deepcopy(configs[arm])
            config.pop("training")
            config["rnn_cell_params"].update(
                hard_reset=reset == "hard", hard_reset_gradient_mode="exact", train_recurrent=False,
            )
            config["inputs"]["replay"]["trainable"] = False
            for edge in config["networks"]["edges"]:
                filename = Path(edge["edges_file"]).name
                edge["edges_file"] = f"output_train_{arm}/callbacks/trained_weights/{filename}"
            folder = Path(f"output_eval_{arm}_{reset}")
            config["output"]["output_dir"] = str(folder)
            path = Path(f"config.eval.{arm}.{reset}.json")
            path.write_text(json.dumps(config, indent=2) + "\n")
            subprocess.run([sys.executable, "run_dpointnet.py", str(path)], check=True)
            spikes = read_spikes(folder, 1.0, 300, 300)
            read_voltages(folder, 1.0, 300, 300)
            events[arm, reset] = spikes
            evaluations[arm][reset] = {
                "rate_mse_hz_squared": float(np.mean((spikes.sum(axis=0) / 0.3 - 30.0) ** 2)),
                "mean_rate_hz": float(spikes.sum() / 90.0),
                "spikes": int(spikes.sum()),
            }
        intended_reset = "hard" if arm == "hard_surrogate" else "soft"
        np.testing.assert_array_equal(
            events[arm, intended_reset], read_spikes(Path(f"output_train_{arm}"), 1.0, 300, 300),
        )
        np.testing.assert_allclose(
            evaluations[arm][intended_reset]["rate_mse_hz_squared"],
            histories[arm][1][-1], rtol=1e-6, atol=1e-6,
        )
    if hashes != input_hashes():
        raise ValueError("Comparison modified shared assets.")
    report = {
        "matched_training_recipes": True, "cpu_only": True,
        "optimizer_updates_per_arm": 6, "input_sha256": hashes,
        "histories": {arm: {"before_updates": values[0], "after_epochs": values[1]} for arm, values in histories.items()},
        "initial_gradient_probes": probes, "initial_weight_gradient_comparison": gradient_comparison,
        "cross_reset_evaluation": evaluations,
        "reset_event_agreement_at_trained_weights": {
            arm: difference(events[arm, "hard"], events[arm, "soft"], 300.0) for arm in ARMS
        },
        "cross_training_event_agreement": {
            reset: difference(events[ARMS[0], reset], events[ARMS[1], reset], 300.0)
            for reset in ("hard", "soft")
        },
        "exported_weights_reproduce_intended_reset_spikes": True,
        "scientific_scope": "Six updates on one replayed toy input; no held-out or convergence qualification",
    }
    output = Path("output_training_comparison")
    output.mkdir(exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in ARMS:
        steps, validation = histories[arm]
        line, = axes[0, 0].plot(np.arange(6), steps, "o-", label=LABELS[arm])
        axes[0, 0].plot([3, 6], validation, "s--", color=line.get_color())
    axes[0, 0].set(xlabel="Optimizer updates", ylabel="Rate MSE (Hz^2)", title="Matched learning: circles before updates; squares after epochs")
    axes[0, 0].legend()
    matrix = np.array([[evaluations[arm][reset]["rate_mse_hz_squared"] for reset in ("hard", "soft")] for arm in ARMS])
    axes[0, 1].imshow(matrix, cmap="YlOrRd")
    for row in range(2):
        for column in range(2):
            axes[0, 1].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axes[0, 1].set(
        xticks=[0, 1], xticklabels=["Hard reset", "Soft reset"],
        yticks=[0, 1], yticklabels=[LABELS[arm] for arm in ARMS],
        xlabel="Inference policy", ylabel="Training policy", title="Exported-weight rate MSE (Hz^2)",
    )
    for axis, arm, reset in ((axes[1, 0], "hard_surrogate", "hard"), (axes[1, 1], "soft", "soft")):
        times, ids = np.nonzero(events[arm, reset])
        axis.scatter(times + 1, ids, s=2)
        axis.set(
            xlabel="Time (ms)", ylabel="Neuron", xlim=(0, 300), ylim=(-1, 300),
            title=f"{LABELS[arm]} after six updates: {len(ids)} spikes",
        )
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare matched hard-surrogate and soft-reset training.")
    parser.add_argument("--probe", choices=ARMS, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.probe:
        probe = initial_gradients(arguments.probe)
        folder = Path(f"output_probe_{arguments.probe}")
        np.savez(folder / "weight_gradients.npz", **probe.pop("weight_gradients"))
        (folder / "initial_gradients.json").write_text(json.dumps(probe, indent=2, allow_nan=False) + "\n")
    else:
        compare()

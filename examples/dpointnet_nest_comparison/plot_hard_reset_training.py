import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_output import read_spikes


def plot():
    probe = json.loads(Path("output_gradient_probe/gradient_probe.json").read_text())
    with Path("output_train_hard_surrogate/callbacks/losses.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["loss_function"] == "__total_loss"]
    steps = [float(row["loss_value"]) for row in rows if row["loss_type"] == "step"]
    validation = [float(row["loss_value"]) for row in rows if row["loss_type"] == "validation"]
    if len(steps) != 6 or len(validation) != 2 or not np.isfinite(steps + validation).all():
        raise ValueError("Expected the complete finite two-epoch, six-update demo.")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(np.arange(6), steps, "o-", label="Before each update")
    axes[0, 0].plot([3, 6], validation, "s--", label="After each epoch")
    axes[0, 0].set(xlabel="Optimizer updates", ylabel="Rate MSE (Hz^2)", title="Hard-forward training")
    axes[0, 0].legend()
    measurements = probe["measurements"]
    axes[0, 1].bar(
        ["Exact hard derivative", "Soft-surrogate derivative"],
        [measurements[mode]["initial_voltage_gradient_l2"] for mode in ("exact", "soft_surrogate")],
    )
    axes[0, 1].set(ylabel="Initial-voltage gradient L2", title="Credit through forced reset and refractory step")
    for axis, folder, label in (
        (axes[1, 0], "output_fp32_hard", "Before training"),
        (axes[1, 1], "output_train_hard_surrogate", "After six updates"),
    ):
        spikes = read_spikes(Path(folder), 1.0, 300, 300)
        times, ids = np.nonzero(spikes)
        axis.scatter(times + 1, ids, s=2)
        axis.set(
            xlabel="Time (ms)", ylabel="Neuron",
            title=f"{label}: {int(spikes.sum())} spikes",
            xlim=(0, 300), ylim=(-1, 300),
        )
    output = Path("output_gradient_probe/hard_reset_training.png")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    plot()

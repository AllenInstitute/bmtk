import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROFILES = ("fp32_hard", "fp32_soft", "mixed_hard", "mixed_soft")
LABELS = {
    "nest": "NEST (double, hard reset)",
    "fp32_hard": "FP32, hard reset",
    "fp32_soft": "FP32, soft reset",
    "mixed_hard": "Mixed, hard reset",
    "mixed_soft": "Mixed, soft reset",
}


def read_spikes(folder, dt, steps, neurons):
    with h5py.File(folder / "spikes.h5") as handle:
        group = handle["spikes/glifs"]
        ids = group["node_ids"][:]
        times = group["timestamps"][:]
    if not np.isfinite(times).all() or np.any(ids < 0) or np.any(ids >= neurons):
        raise ValueError(f"Invalid spike data in {folder}")
    bins = np.rint(times / dt).astype(int) - 1
    if np.any(bins < 0) or np.any(bins >= steps):
        raise ValueError(f"Spike outside the simulation end-of-step grid in {folder}")
    if not np.allclose(times, (bins + 1) * dt, rtol=0, atol=1e-7):
        raise ValueError(f"Off-grid spike timestamps in {folder}")
    spikes = np.zeros((steps, neurons), dtype=bool)
    spikes[bins, ids] = True
    if int(spikes.sum()) != len(ids):
        raise ValueError(f"Duplicate neuron/timestep events in {folder}")
    return spikes


def read_voltages(folder, dt, steps, neurons):
    with h5py.File(folder / "voltages.h5") as handle:
        group = handle["report/glifs"]
        data = group["data"][:]
        node_ids = group["mapping/node_id"][:]
        time = group["mapping/time"][:]
    if data.shape != (steps, neurons) or not np.isfinite(data).all():
        raise ValueError(f"Invalid voltage data in {folder}")
    if not np.array_equal(node_ids, np.arange(neurons)):
        raise ValueError(f"Unexpected voltage column ordering in {folder}")
    if not np.allclose(time, [dt, (steps + 1) * dt, dt], rtol=0, atol=1e-7):
        raise ValueError(f"Unexpected voltage time mapping in {folder}")
    return data


def difference(left, right, duration):
    common = int(np.count_nonzero(left & right))
    left_count = int(left.sum())
    right_count = int(right.sum())
    return {
        "left_spikes": left_count,
        "right_spikes": right_count,
        "matched_neuron_time_events": common,
        "symmetric_difference_events": int(np.count_nonzero(left ^ right)),
        "event_f1": 2.0 * common / (left_count + right_count)
        if left_count + right_count else 1.0,
        "mean_absolute_rate_difference_hz": float(
            np.mean(np.abs(
                left.sum(axis=0).astype(np.int64) - right.sum(axis=0).astype(np.int64)
            )) * 1000.0 / duration
        ),
    }


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(without_nest=False, node_id=200, output="output_comparison"):
    config = json.loads(Path("config.base.json").read_text())
    dt = config["run"]["dt"]
    steps = config["run"]["seq_len"]
    duration = dt * steps
    build = json.loads(Path("network/build.json").read_text())
    neurons = build["neurons"]
    if build["dt_ms"] != dt or build["duration_ms"] != duration:
        raise ValueError("Build/input grid differs from the simulation configuration")
    if not 0 <= node_id < neurons:
        raise ValueError("node-id must select a model neuron")
    names = list(PROFILES) if without_nest else ["nest", *PROFILES]
    spikes = {}
    reports = {}
    voltages = {}
    for name in names:
        folder = Path(f"output_{name}")
        reports[name] = json.loads((folder / "resolved.json").read_text())
        if reports[name]["dt_ms"] != dt:
            raise ValueError(f"Different timestep in {folder}")
        if reports[name]["input_sha256"] != reports[names[0]]["input_sha256"]:
            raise ValueError(f"Network/components/replayed input differ in {folder}")
        spikes[name] = read_spikes(folder, dt, steps, neurons)
        if name != "nest":
            voltages[name] = read_voltages(folder, dt, steps, neurons)

    baseline = "fp32_hard" if without_nest else "nest"
    comparison = [
        {
            "profile": name,
            "reference": baseline,
            "spikes": int(spikes[name].sum()),
            "mean_rate_hz": float(spikes[name].mean() * 1000.0 / dt),
            **difference(spikes[name], spikes[baseline], duration),
        }
        for name in names
    ]
    pairwise = []
    for left, right, effect in (
        ("fp32_hard", "fp32_soft", "Reset at FP32"),
        ("fp32_hard", "mixed_hard", "Precision at hard reset"),
        ("fp32_soft", "mixed_soft", "Precision at soft reset"),
        ("mixed_hard", "mixed_soft", "Reset at mixed precision"),
    ):
        pairwise.append(
            {
                "effect": effect,
                "left": left,
                "right": right,
                **difference(spikes[left], spikes[right], duration),
                "normalized_voltage_rms_difference": float(
                    np.sqrt(np.mean(np.square(
                        voltages[left].astype(np.float64) - voltages[right]
                    )))
                ),
            }
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "comparison.csv", comparison)
    write_table(output / "pairwise.csv", pairwise)
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "dt_ms": dt,
                "duration_ms": duration,
                "neurons": neurons,
                "reference": baseline,
                "time_shift_applied": False,
                "comparison": comparison,
                "pairwise": pairwise,
                "resolved": reports,
            },
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    figure, axes = plt.subplots(
        len(names) + 2, 1, figsize=(12, 2 * len(names) + 5),
        sharex=True, constrained_layout=True,
    )
    colors = dict(zip(names, plt.get_cmap("tab10").colors))
    time = (np.arange(steps) + 1) * dt
    for axis, name in zip(axes, names):
        rows, columns = np.nonzero(spikes[name])
        axis.scatter(time[rows], columns, s=2, color=colors[name])
        axis.set_ylabel("Neuron")
        axis.set_title(f"{LABELS[name]}: {int(spikes[name].sum())} spikes", loc="left")
        axis.set_ylim(-1, neurons)
    for name in PROFILES:
        axes[-2].plot(time, voltages[name][:, node_id], label=LABELS[name], color=colors[name])
        axes[-1].plot(
            time, np.cumsum(np.count_nonzero(spikes[name] ^ spikes[baseline], axis=1)),
            label=LABELS[name], color=colors[name],
        )
    axes[-2].axhline(1.0, color="gray", linestyle=":", label="Spike threshold")
    axes[-2].set_ylabel("Normalized voltage")
    axes[-2].set_title(f"Neuron {node_id}: post-reset voltage (threshold = 1)", loc="left")
    axes[-2].legend(ncol=3, fontsize=8)
    axes[-1].set_ylabel("Differing events")
    axes[-1].set_title(f"Cumulative neuron/time-bin disagreements vs {LABELS[baseline]}", loc="left")
    axes[-1].set_xlabel("Time (ms)")
    axes[-1].legend(ncol=2, fontsize=8)
    axes[-1].set_xlim(0, duration)
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)
    print(f"Saved plots and comparison tables in {output.resolve()}")
    for row in comparison:
        print(f'{row["profile"]}: {row["spikes"]} spikes, event F1={row["event_f1"]:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare all matched reset/precision runs.")
    parser.add_argument("--without-nest", action="store_true", help="Explicitly compare only the four DPointNet profiles.")
    parser.add_argument("--node-id", type=int, default=200)
    parser.add_argument("--output-dir", default="output_comparison")
    args = parser.parse_args()
    plot(args.without_nest, args.node_id, args.output_dir)

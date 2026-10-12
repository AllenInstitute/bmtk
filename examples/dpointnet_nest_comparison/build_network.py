import argparse
import json
from pathlib import Path

import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.utils.reports.spike_trains import SpikeTrains


DT = 1.0
DURATION = 300.0


def build(seed=3000):
    rng = np.random.default_rng(seed)
    recurrent = NetworkBuilder("glifs")
    for count, kind, filename in (
        (200, "e", "excitatory.json"),
        (100, "i", "inhibitory.json"),
    ):
        recurrent.add_nodes(
            N=count,
            ei=kind,
            model_type="point_neuron",
            model_template="nest:glif_psc",
            dynamics_params=filename,
        )

    for kind, component, low, high in (
        ("e", "excitatory.json", 1.0, 4.0),
        ("i", "inhibitory.json", -8.0, -2.0),
    ):
        edges = recurrent.add_edges(
            source={"ei": kind},
            target=recurrent.nodes(),
            connection_rule=lambda source, target: int(source.node_id != target.node_id),
            dynamics_params=component,
            model_template="static_synapse",
            delay=1.0,
        )
        edges.add_properties(
            "syn_weight", rule=lambda *_, lower=low, upper=high: rng.uniform(lower, upper)
        )
    recurrent.build()
    recurrent.save("network")

    inputs = NetworkBuilder("virts")
    inputs.add_nodes(N=100, model_type="virtual")
    for kind, low, high in (("e", 35.0, 55.0), ("i", 15.0, 25.0)):
        edges = inputs.add_edges(
            target=recurrent.nodes(ei=kind),
            connection_rule=1,
            dynamics_params="excitatory.json",
            model_template="static_synapse",
            delay=1.0,
        )
        edges.add_properties(
            "syn_weight", rule=lambda *_, lower=low, upper=high: rng.uniform(lower, upper)
        )
    inputs.build()
    inputs.save("network")

    steps = np.arange(20, int(DURATION / DT) - 20)
    probability = 20.0 * DT / 1000.0
    active = rng.random((len(steps), 100)) < probability
    rows, columns = np.nonzero(active)
    spikes = SpikeTrains()
    spikes.add_spikes(node_ids=columns, timestamps=steps[rows] * DT, population="virts")
    Path("inputs").mkdir(exist_ok=True)
    spikes.to_sonata("inputs/spikes.h5")
    Path("network/build.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "neurons": 300,
                "input_nodes": 100,
                "dt_ms": DT,
                "duration_ms": DURATION,
                "input_events": len(rows),
                "input_sampling": "Seeded grid Bernoulli events; identical saved realization in all arms",
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the matched 300-neuron GLIF comparison.")
    parser.add_argument("--seed", type=int, default=3000)
    build(parser.parse_args().seed)

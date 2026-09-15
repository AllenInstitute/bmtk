from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.input_modules.delayed_cue_spikes import (
    DelayedCueSpikes,
    generate_delayed_cue_spikes,
)
from bmtk.simulator.dpointnet.loss_functions.delayed_association import (
    DelayedAssociationLoss,
)


def test_delayed_cues_are_balanced_temporal_and_reproducible():
    def module():
        return DelayedCueSpikes(
            rnn=SimpleNamespace(adjusted_seq_len=170, dt=1.0),
            name="cues",
            input_network=SimpleNamespace(name="virts", n_nodes=100, input_type=None),
            delays_ms=[25.0, 75.0],
            cue_duration_ms=20.0,
            background_rate_hz=0.0,
            cue_rate_hz=1000.0,
            seed=17,
        )

    first_iterator = iter(module().create_generator())
    second_iterator = iter(module().create_generator())
    first = [next(first_iterator) for _ in range(4)]
    second = [next(second_iterator) for _ in range(4)]

    assert sorted((int(y["class_label"]), float(y["delay_ms"])) for _, y in first) == [
        (0, 25.0),
        (0, 75.0),
        (1, 25.0),
        (1, 75.0),
    ]
    for (spikes, target), (other_spikes, other_target) in zip(first, second):
        np.testing.assert_array_equal(spikes.numpy(), other_spikes.numpy())
        assert int(target["class_label"]) == int(other_target["class_label"])
        label = int(target["class_label"])
        active = slice(0, 50) if label == 0 else slice(50, 100)
        inactive = slice(50, 100) if label == 0 else slice(0, 50)
        assert tf.reduce_sum(spikes[:20, active]).numpy() > 0
        assert tf.reduce_sum(spikes[:, inactive]).numpy() == 0
        assert tf.reduce_sum(spikes[20:]).numpy() == 0


def test_delayed_loss_prefers_matching_fixed_readout_pool():
    loss = DelayedAssociationLoss(
        rnn=SimpleNamespace(dt=1.0),
        pool_a_start=0,
        pool_a_end=2,
        pool_b_start=2,
        pool_b_end=4,
        cue_duration_ms=2.0,
        response_window_ms=2.0,
        temperature_hz=5.0,
    )
    matching = tf.constant(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        ],
        dtype=tf.float32,
    )
    targets = [
        [
            {
                "class_label": tf.constant([0, 1]),
                "delay_ms": tf.constant([0.0, 0.0]),
            }
        ]
    ]

    matching_loss = loss(matching, targets)
    swapped_loss = loss(tf.reverse(matching, axis=[2]), targets)

    assert matching_loss.numpy() < swapped_loss.numpy()
    assert matching_loss.numpy() < 0.01


def test_response_probe_is_class_neutral_and_delay_aligned():
    spikes = generate_delayed_cue_spikes(
        rng=np.random.default_rng(19),
        seq_len=100,
        n_nodes=100,
        dt_ms=1.0,
        label=0,
        delay_ms=25.0,
        cue_duration_ms=20.0,
        background_rate_hz=0.0,
        cue_rate_hz=1000.0,
        probe_duration_ms=10.0,
        probe_rate_hz=1000.0,
    )

    assert spikes[20:45].sum() == 0
    assert spikes[45:55, :50].sum() > 0
    assert spikes[45:55, 50:].sum() > 0
    assert spikes[55:].sum() == 0

from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.input_modules.poisson_spikes import PoissonSpikes


def _module(firing_rate, seed=17):
    rnn = SimpleNamespace(adjusted_seq_len=20)
    network = SimpleNamespace(name="input", n_nodes=3, input_type=None)
    return PoissonSpikes(
        rnn=rnn,
        name="poisson",
        input_network=network,
        firing_rate=firing_rate,
        seed=seed,
    )


def test_poisson_spikes_samples_configured_rates_reproducibly():
    first = iter(_module([0.0, 1000.0]).create_generator())
    second = iter(_module([0.0, 1000.0]).create_generator())
    first_samples = [next(first) for _ in range(12)]
    second_samples = [next(second) for _ in range(12)]

    first_rates = [
        float(metadata["firing_rate"].numpy()) for _, metadata in first_samples
    ]
    second_rates = [
        float(metadata["firing_rate"].numpy()) for _, metadata in second_samples
    ]
    assert first_rates == second_rates
    assert set(first_rates) == {0.0, 1000.0}
    for (first_spikes, _), (second_spikes, _) in zip(first_samples, second_samples):
        np.testing.assert_array_equal(first_spikes.numpy(), second_spikes.numpy())


@pytest.mark.parametrize("firing_rate", [[], [-1.0], [[1.0, 2.0]]])
def test_poisson_spikes_rejects_invalid_rate_sets(firing_rate):
    with pytest.raises(ValueError, match="firing_rate"):
        _module(firing_rate)

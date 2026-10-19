"""Exact CPU Poisson preparation must preserve draws, state and replay credit."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("start,seed", [(0, 17), (12, 29)])
@pytest.mark.parametrize("two_inputs", [False, True])
def test_prepared_counts_exact_live_seed_and_step(selective, start, seed, two_inputs):
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import (
        _prepare_cpu_poisson, _prepared_noise_step, _rollout_context,
    )

    cell = make_cell(
        "nest", selective=selective, noise=True, two_inputs=two_inputs,
        use_prepared_cpu_poisson=True,
    )
    cell.noise_seed.assign(seed)
    states = list(cell.zero_state(3, cell.compute_dtype))
    states[6] = tf.fill((3,), start)
    inputs = tf.zeros((3, 7, 0), tf.as_dtype(cell.compute_dtype))
    try:
        with _rollout_context(cell):
            tables = _prepare_cpu_poisson(cell, inputs, states)
            for net in cell.inputs.values():
                for step in range(7):
                    expected = cell.sample_noise_spikes(3, states[6] + step, net)
                    with _prepared_noise_step(cell, tables, tf.constant(step, tf.int64)):
                        actual = cell.sample_noise_spikes(3, states[6] + step, net)
                    assert actual.dtype == tf.int32
                    np.testing.assert_array_equal(actual, expected)
                assert tables[id(net)].dtype == tf.int64
                assert np.any(tables[id(net)].numpy() > 1), "Poisson counts must not become Bernoulli"
                if tf.config.list_physical_devices("GPU"):
                    assert "GPU:" in tables[id(net)].device
                    assert "GPU:" in actual.device
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_prepared_loop_preserves_values_gradients_and_remainder(selective, segmented, native, tmp_path):
    from test_precision_credit import make_cell, build_core
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    if native:
        from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops
        if not csr_spike_ops.fused_recurrent_accumulation_available():
            pytest.skip("Native accumulation unavailable")
    cell = make_cell(
        "nest", selective=selective, noise=True, fused=native,
        use_direct_state_rnn_loop=True, use_fused_cuda=native,
        use_fused_recurrent_accumulation=native, _canonical_gradient_boundary=native,
        use_direct_csr_recurrent_gradient=native,
        use_pair_projection=native,
    )
    try:
        core, states, sequence = build_core(cell, length=7, batch=3)
        targets = [cell.recurrent_weight_values, cell.inputs["drive"]["input_weight_values"]]

        def build():
            runner = SegmentedRecomputeRunner(core, 7, 3, 2, differentiate_inputs=False) if segmented else None
            @tf.function
            def evaluate():
                with tf.GradientTape() as tape:
                    outputs = runner(sequence, states) if segmented else core([sequence, *states])
                    objective = tf.add_n([
                        tf.reduce_sum(tf.cast(value, tf.float32) * tf.random.stateless_normal(
                            tf.shape(value), [31, index],
                        )) for index, value in enumerate(outputs) if value.dtype.is_floating
                    ])
                return outputs, tape.gradient(objective, targets)
            return evaluate

        baseline = build()
        expected = baseline()
        cell._use_prepared_cpu_poisson = True
        # Rebuild the functional model so the explicit opt-in is traced, not cached.
        core, states, sequence = build_core(cell, length=7, batch=3)
        candidate = build()
        actual = candidate()
        assert all(value is not None for value in actual[1])
        assert all(np.linalg.norm(value) > 0 for value in actual[1])
        for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
            np.testing.assert_array_equal(left, right)
        checkpoint = tf.train.Checkpoint(noise_seed=cell.noise_seed)
        saved = checkpoint.write(str(tmp_path / "noise"))
        cell.noise_seed.assign(37)
        cell._use_prepared_cpu_poisson = False
        refreshed_baseline = baseline()
        cell._use_prepared_cpu_poisson = True
        refreshed_candidate = candidate()
        for left, right in zip(tf.nest.flatten(refreshed_candidate), tf.nest.flatten(refreshed_baseline)):
            np.testing.assert_array_equal(left, right)
        checkpoint.read(saved).assert_consumed()
        restored = candidate()
        for left, right in zip(tf.nest.flatten(restored), tf.nest.flatten(actual)):
            np.testing.assert_array_equal(left, right)
    finally:
        cell.close_fused_cuda()


def test_context_restored_on_nested_error_and_seed_override():
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import (
        _prepared_noise_step, _prepared_noise_for,
    )
    cell = make_cell("nest", noise=True, use_prepared_cpu_poisson=True)
    net = cell.inputs["drive"]
    tables = {id(net): tf.ones((2, 3, 2), tf.int64)}
    try:
        assert _prepared_noise_for(cell, net) is None
        with _prepared_noise_step(cell, tables, 0):
            with pytest.raises(ValueError, match="overridden"):
                cell.sample_noise_spikes(3, tf.zeros((3,), tf.int32), net, noise_seed=17)
            with pytest.raises(RuntimeError):
                with _prepared_noise_step(cell, tables, 1):
                    raise RuntimeError("test cleanup")
            np.testing.assert_array_equal(_prepared_noise_for(cell, net), np.ones((3, 2)))
        assert _prepared_noise_for(cell, net) is None
    finally:
        cell.close_fused_cuda()


def test_prepared_budget_and_boolean_conflict():
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import _prepare_cpu_poisson
    with pytest.raises(ValueError, match="true or false"):
        make_cell("nest", use_prepared_cpu_poisson="true")
    with pytest.raises(ValueError, match="device Poisson"):
        make_cell("nest", use_prepared_cpu_poisson=True, use_device_poisson=True)
    cell = make_cell("nest", noise=True, use_prepared_cpu_poisson=True)
    try:
        cell.inputs["drive"]["input_dim"] = 64 * 1024**2
        with pytest.raises(tf.errors.InvalidArgumentError, match="budget"):
            _prepare_cpu_poisson(cell, tf.zeros((3, 1, 0)), cell.zero_state(3, cell.compute_dtype))
    finally:
        cell.close_fused_cuda()


def test_b32_t500_background_count_hashes_match():
    import hashlib
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import (
        _prepare_cpu_poisson, _prepared_noise_step, _rollout_context,
    )
    cell = make_cell("nest", noise=True, use_prepared_cpu_poisson=True)
    net = cell.inputs["drive"]
    net["input_dim"] = 100
    states = cell.zero_state(32, cell.compute_dtype)
    try:
        @tf.function
        def evaluate():
            with _rollout_context(cell):
                tables = _prepare_cpu_poisson(cell, tf.zeros((32, 500, 0)), states)
                expected = tf.map_fn(
                    lambda index: cell.sample_noise_spikes(32, states[6] + index, net),
                    tf.range(500), fn_output_signature=tf.TensorSpec((32, 100), tf.int32),
                )
                def prepared(index):
                    with _prepared_noise_step(cell, tables, tf.cast(index, tf.int64)):
                        return cell.sample_noise_spikes(32, states[6] + index, net)
                actual = tf.map_fn(
                    prepared, tf.range(500), fn_output_signature=tf.TensorSpec((32, 100), tf.int32),
                )
                return actual, expected
        for seed in (17, 37):
            cell.noise_seed.assign(seed)
            actual, expected = evaluate()
            left, right = actual.numpy(), expected.numpy()
            assert hashlib.sha256(left.tobytes()).hexdigest() == hashlib.sha256(right.tobytes()).hexdigest()
            np.testing.assert_array_equal(left, right)
    finally:
        cell.close_fused_cuda()

"""Host input formatting preserves model casts, counts and canonical metadata."""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.data_iterator import DataIterator
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops


@pytest.mark.parametrize("dtype", [tf.float16, tf.bfloat16, tf.float32, tf.float64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("inputs", ["binary", "counts", "mixed_float"])
def test_output_dtype_matches_post_concat_model_cast(dtype, compiled, inputs):
    if inputs == "binary":
        values = [tf.constant([[[True, False]]]), tf.constant([[[False, True]]])]
    elif inputs == "counts":
        values = [tf.constant([[[True, False]]]), tf.constant([[[2.0, 7.0]]])]
    else:
        values = [tf.constant([[[0.1, 0.3]]], tf.float16), tf.constant([[[0.1234567]]])]
    expected = tf.cast(DataIterator._concat_spikes(values), dtype)

    def combine():
        return DataIterator._concat_spikes(values, output_dtype=dtype)

    actual = tf.function(combine)() if compiled else combine()
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)
    if inputs == "counts":
        assert float(actual[0, 0, -1]) == 7


@pytest.mark.parametrize("listed", [False, True])
def test_iterator_formats_output_but_preserves_targets(listed):
    iterator = DataIterator([], 1, 1, output_dtype=tf.float16)
    target = tf.constant([17.0])
    groups = [
        iter([(tf.constant([[[True, False]]]), target)]),
        iter([(tf.constant([[[2.0, 7.0]]]), target)]),
    ]
    iterator.data_itrs = [groups] if listed else groups
    spikes, targets = iterator._next_spikes_list() if listed else iterator._next_spikes()
    if listed:
        spikes, targets = spikes[0], targets[0]
    assert spikes.dtype == tf.float16
    np.testing.assert_array_equal(spikes, [[[1, 0, 2, 7]]])
    assert all(value is target for value in targets)
    assert DataIterator([], 1, 1).output_dtype is None


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_distributed_output_dtype_on_actual_worker(dtype):
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("Actual GPU placement required")
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    values = strategy.experimental_distribute_values_from_function(
        lambda _: tf.constant([[[2.0, 7.0]]])
    )
    actual = DataIterator._concat_spikes([values], strategy, dtype)
    local = strategy.experimental_local_results(actual)[0]
    assert "GPU:" in local.device
    assert local.dtype == dtype
    np.testing.assert_array_equal(local, [[[2.0, 7.0]]])


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("selective", [False, True])
def test_canonical_type_values_dtype_placement_and_index_variables(combined, continuous, selective):
    if not csr_spike_ops.combined_sweep_available():
        pytest.skip("Actual native CUDA required")
    from test_precision_credit import make_cell

    policy = tf.keras.mixed_precision.global_policy()
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    cell = None
    try:
        with strategy.scope(), tf.device("/GPU:0"):
            def create():
                return make_cell(
                    "nest", selective=selective, noise=not continuous, batch_size=32,
                    fused=True, use_fused_cuda=True, use_pair_projection=True,
                    use_combined_current_sweep=combined,
                )
            if combined and continuous:
                with pytest.raises(ValueError, match="native spike-input"):
                    create()
                return
            cell = create()
        assert cell.syn_ids.dtype == tf.int64
        np.testing.assert_array_equal(cell.syn_ids, [0, 0])
        assert ("CPU:" in cell.syn_ids.device) == combined
        assert isinstance(cell.recurrent_indices, tf.Variable)
        assert cell.recurrent_indices.dtype == tf.int64
        for network in cell.inputs.values():
            assert network["input_syn_ids"].dtype == tf.int64
            np.testing.assert_array_equal(network["input_syn_ids"], [0, 0])
            assert ("CPU:" in network["input_syn_ids"].device) == (combined and not continuous)
            assert isinstance(network["input_indices"], tf.Variable)
            assert network["input_indices"].dtype == tf.int64
    finally:
        if cell is not None:
            cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("selective", [False, True])
def test_host_index_variables_native_vjps_and_cross_placement_checkpoint(selective, tmp_path):
    if not csr_spike_ops.combined_sweep_available():
        pytest.skip("Actual native CUDA required")
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
        GLIF3Cell, _prepare_host_index_variables,
    )

    policy = tf.keras.mixed_precision.global_policy()
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0"])
    cells = []
    try:
        network, inputs, options = make_cell(
            "nest", selective=selective, noise=True, batch_size=32,
            fused=True, use_fused_cuda=True, use_pair_projection=True,
            use_combined_current_sweep=True, return_spec=True,
        )
        prepared = _prepare_host_index_variables(network, inputs)
        with strategy.scope(), tf.device("/GPU:0"):
            cells.append(GLIF3Cell(network, inputs, **options))
            cells.append(GLIF3Cell(network, inputs, _host_index_variables=prepared, **options))
        old, new = cells
        assert "GPU:" in old.recurrent_indices.device
        assert "CPU:" in new.recurrent_indices.device
        np.testing.assert_array_equal(new.recurrent_indices, old.recurrent_indices)
        for name in new.inputs:
            assert "CPU:" in new.inputs[name]["input_indices"].device
            np.testing.assert_array_equal(
                new.inputs[name]["input_indices"], old.inputs[name]["input_indices"]
            )
        results = []
        for cell in cells:
            def calculate():
                spikes = tf.ones((32, cell.recurrent_dense_shape[1]), cell.compute_dtype)
                with tf.GradientTape() as tape:
                    tape.watch(spikes)
                    currents = cell.calculate_i_rec_with_custom_grad(spikes)
                    loss = tf.reduce_sum(tf.cast(currents, tf.float32))
                gradients = tape.gradient(loss, [spikes, cell.recurrent_weight_values])
                assert all(value is not None for value in gradients)
                return currents, *gradients
            observed = strategy.run(calculate)
            results.append(tuple(strategy.experimental_local_results(value)[0] for value in observed))
        for before, after in zip(*results):
            np.testing.assert_array_equal(before, after)

        def checkpoint(cell):
            return tf.train.Checkpoint(
                cell=cell,
                **{f"input_index_{index}": item["input_indices"]
                   for index, item in enumerate(cell.inputs.values())},
            )

        old_schema_path = tf.train.Checkpoint(cell=old).save(str(tmp_path / "old-schema"))
        new_schema_path = tf.train.Checkpoint(cell=new).save(str(tmp_path / "new-schema"))
        assert tf.train.list_variables(old_schema_path) == tf.train.list_variables(new_schema_path)
        path = checkpoint(old).save(str(tmp_path / "gpu"))
        new.recurrent_indices.assign(tf.zeros_like(new.recurrent_indices))
        for item in new.inputs.values():
            item["input_indices"].assign(tf.zeros_like(item["input_indices"]))
        checkpoint(new).restore(path).assert_consumed()
        np.testing.assert_array_equal(new.recurrent_indices, old.recurrent_indices)
        for name in new.inputs:
            np.testing.assert_array_equal(
                new.inputs[name]["input_indices"], old.inputs[name]["input_indices"]
            )
        path = checkpoint(new).save(str(tmp_path / "cpu"))
        old.recurrent_indices.assign(tf.zeros_like(old.recurrent_indices))
        checkpoint(old).restore(path).assert_consumed()
        np.testing.assert_array_equal(old.recurrent_indices, new.recurrent_indices)
    finally:
        for cell in cells:
            cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("bad", ["shape", "dtype", "trainable"])
def test_prepared_index_contract_rejects_invalid_variables(bad):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import _canonical_index_variable

    with tf.device("/CPU:0"):
        value = tf.Variable(
            [[0]] if bad == "shape" else [[0, 1]],
            dtype=tf.float32 if bad == "dtype" else tf.int64,
            trainable=bad == "trainable",
        )
    with pytest.raises(ValueError, match="matching nontrainable CPU int64"):
        _canonical_index_variable(np.array([[0, 1]], np.int64), value)

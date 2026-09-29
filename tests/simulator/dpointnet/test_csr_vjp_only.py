"""Projection-free replay retains the registered sparse VJP, including resource captures."""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity, fused_cuda_available, fused_spike_currents,
    reorder_csr_values, restore_csr_values,
)
from test_csr_fp32_batch32 import _fixture, _oracle, _assert_vjp


@pytest.mark.parametrize("direct,recurrent", [(False, True), (True, True), (False, False)])
def test_vjp_only_registered_gradient_matches_independent_oracle(direct, recurrent):
    if not fused_cuda_available():
        pytest.skip("Rebuilt CUDA required")
    fixture = _fixture("counts")
    indices, types, weights, basis, upstream, spikes = fixture
    _, expected_ds, expected_dw, ds_bound, dw_bound = _oracle(fixture, direct, .5)
    connectivity = build_csr_connectivity(
        indices, types, spikes.shape[1], upstream.shape[1], len(basis),
        build_compact_pairs=True,
    )
    try:
        with tf.device("/GPU:0"):
            source = tf.Variable(spikes)
            master = tf.Variable(weights)
            csr = reorder_csr_values(master, connectivity)

            @tf.function
            def calculate():
                with tf.GradientTape() as tape:
                    dummy = fused_spike_currents(
                        source, master, csr, connectivity, tf.constant(basis),
                        upstream.shape[1], recurrent, spike_gradient_scale=.5,
                        use_packed_sm120_backward=False,
                        write_csr_weight_gradient=direct, vjp_only=True,
                    )
                    objective = tf.reduce_sum(dummy * upstream.reshape(-1, 4))
                ds, dw = tape.gradient(objective, (source, master))
                return dummy, ds, restore_csr_values(dw, connectivity) if direct else dw
            dummy, ds, dw = calculate()
        np.testing.assert_array_equal(dummy, 0)
        if recurrent:
            _assert_vjp(ds.numpy(), expected_ds, ds_bound)
        else:
            assert ds is None
        _assert_vjp(dw.numpy(), expected_dw, dw_bound)
    finally:
        connectivity.close()

"""Projection-free replay retains the registered sparse VJP, including resource captures."""

import numpy as np
import pytest
import tensorflow as tf
from contextlib import nullcontext

from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity, fused_cuda_available, fused_spike_currents,
    reorder_csr_values, restore_csr_values,
)
from test_csr_fp32_batch32 import _fixture, _oracle, _assert_vjp
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops


def test_elided_projection_scope_restores_nested_and_exceptional_entries():
    assert not csr_spike_ops._ELIDE_VJP_PRODUCERS.get()
    with csr_spike_ops._elide_vjp_producers():
        assert csr_spike_ops._ELIDE_VJP_PRODUCERS.get()
        with pytest.raises(RuntimeError, match="scope probe"):
            with csr_spike_ops._elide_vjp_producers():
                raise RuntimeError("scope probe")
        assert csr_spike_ops._ELIDE_VJP_PRODUCERS.get()
    assert not csr_spike_ops._ELIDE_VJP_PRODUCERS.get()


@pytest.mark.parametrize("elided", [False, True])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("direct,recurrent", [(False, True), (True, True), (False, False)])
def test_vjp_only_registered_gradient_matches_independent_oracle(direct, recurrent, elided, steps, monkeypatch):
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
            if elided:
                def reject_forward(*args, **kwargs):
                    raise AssertionError("VJP-only projection executed a native forward producer")
                monkeypatch.setattr(csr_spike_ops._OPS, "dpointnet_csr_spike_forward", reject_forward)
                monkeypatch.setattr(csr_spike_ops._OPS, "dpointnet_csr_spike_forward_csr_master", reject_forward)

            @tf.function
            def calculate():
                with tf.GradientTape() as tape:
                    def projection():
                        return fused_spike_currents(
                            source, master, csr, connectivity, tf.constant(basis),
                            upstream.shape[1], recurrent, spike_gradient_scale=.5,
                            use_packed_sm120_backward=False,
                            write_csr_weight_gradient=direct, vjp_only=True,
                        )
                    if steps == 1:
                        dummy = projection()
                    else:
                        dummy = tf.while_loop(
                            lambda index, _: index < steps,
                            lambda index, value: (index + 1, value + projection()),
                            (tf.constant(0), tf.zeros(upstream.reshape(-1, 4).shape, source.dtype)),
                        )[1]
                    objective = tf.reduce_sum(dummy * upstream.reshape(-1, 4))
                ds, dw = tape.gradient(objective, (source, master))
                return dummy, ds, restore_csr_values(dw, connectivity) if direct else dw
            with csr_spike_ops._elide_vjp_producers() if elided else nullcontext():
                dummy, ds, dw = calculate()
        assert dummy.shape == (spikes.shape[0] * upstream.shape[1], basis.shape[1])
        np.testing.assert_array_equal(dummy, 0)
        if recurrent:
            _assert_vjp(ds.numpy(), steps * expected_ds, steps * ds_bound)
        elif steps > 1:
            np.testing.assert_array_equal(ds.numpy(), 0)
        else:
            assert ds is None
        _assert_vjp(dw.numpy(), steps * expected_dw, steps * dw_bound)
    finally:
        connectivity.close()

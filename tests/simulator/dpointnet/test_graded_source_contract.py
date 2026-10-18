"""Independent FP64 oracle for graded CSR values and canonical dense VJPs."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
    build_csr_connectivity,
    fused_cuda_available,
    fused_spike_currents,
    reorder_csr_values,
)


@pytest.mark.parametrize("batch", [1, 4, 32])
@pytest.mark.parametrize("compact_pairs", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_graded_values_dense_silent_credit_and_canonical_weight_order(batch, compact_pairs, compiled):
    if not fused_cuda_available():
        pytest.skip("Requires rebuilt CUDA operators and an authorized GPU")
    indices = np.array([[2, 0], [0, 2], [1, 1], [0, 0], [1, 2], [2, 0]])
    types = np.array([1, 0, 1, 0, 1, 1])
    weights = np.array([0.3, -0.2, 0.4, 0.5, -0.6, 0.7], np.float32)
    basis = np.array([[0.5, 0.2, 1.0, 0.7], [1.0, 0.6, 0.1, 0.3]], np.float32)
    sources = np.tile(np.array([0.25, 0.0, 2.0, 0.5], np.float32), (batch, 1))
    sources[:, 0] += np.arange(batch, dtype=np.float32) / 128
    upstream = np.random.default_rng(20261017).normal(size=(batch, 3, 4)).astype(np.float32)
    currents64 = np.zeros((batch, 3, 4), np.float64)
    source64 = np.zeros(sources.shape, np.float64)
    weight64 = np.zeros(weights.shape, np.float64)
    for edge, (post, pre) in enumerate(indices):
        factors = basis[types[edge]].astype(np.float64)
        credit = upstream[:, post].astype(np.float64) @ factors
        currents64[:, post] += sources[:, pre, None].astype(np.float64) * float(weights[edge]) * factors
        source64[:, pre] += credit * float(weights[edge])
        weight64[edge] = np.dot(credit, sources[:, pre].astype(np.float64))

    connectivity = build_csr_connectivity(
        indices, types, 4, 3, 2, build_compact_pairs=compact_pairs
    )
    try:
        with tf.device("/GPU:0"):
            source = tf.Variable(sources)
            master = tf.Variable(weights)
            csr = reorder_csr_values(master, connectivity)

            def evaluate():
                with tf.GradientTape() as tape:
                    current = fused_spike_currents(
                        source, master, csr, connectivity, tf.constant(basis), 3,
                        compute_spike_gradient=True, write_csr_weight_gradient=False,
                        use_packed_sm120_backward=False,
                    )
                    loss = tf.reduce_sum(current * upstream.reshape(-1, 4))
                return current, *tape.gradient(loss, [source, master])

            current, ds, dw = (tf.function(evaluate) if compiled else evaluate)()
        assert ds.dtype == dw.dtype == tf.float32
        np.testing.assert_allclose(current.numpy().reshape(batch, 3, 4), currents64, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(ds, source64, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(dw, weight64, rtol=2e-5, atol=2e-5)
        assert np.any(ds[:, 1].numpy() != 0), "Silent connected sources retain credit."
        np.testing.assert_array_equal(ds[:, 3], 0)
        np.testing.assert_array_equal(dw[2], 0)
    finally:
        connectivity.close()

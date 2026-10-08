"""Independent CPU oracles and CUDA qualification for the FP32 packed VJP."""

import numpy as np
import pytest


def _fixture(kind, tiny=False):
    rng = np.random.default_rng(260926)
    degrees = np.array([0, 1, 3, 31, 32, 33, 63, 64, 65, 129, 4099, 0])
    pre = np.repeat(np.arange(len(degrees)), degrees)
    post = rng.integers(0, 17, size=len(pre))
    types = rng.integers(0, 5, size=len(pre))
    permutation = rng.permutation(len(pre))
    indices = np.stack([post, pre], axis=1)[permutation]
    types = types[permutation]
    weights = rng.normal(0, 0.3, len(pre)).astype(np.float32)
    basis = rng.normal(0, 0.5, (5, 4)).astype(np.float32)
    upstream = rng.normal(0, 0.4, (32, 17, 4)).astype(np.float32)
    if tiny:
        upstream *= np.float32(1e-9)
    if kind == "binary":
        spikes = (rng.uniform(size=(32, len(degrees))) < 0.15).astype(np.float32)
    elif kind == "counts":
        spikes = rng.poisson(0.6, (32, len(degrees))).astype(np.float32)
    elif kind == "fractional":
        spikes = rng.uniform(0, 2, (32, len(degrees))).astype(np.float32)
    else:
        spikes = rng.normal(size=(32, len(degrees))).astype(np.float32)
        spikes[:, 5] = -abs(spikes[:, 5])
    spikes[:, 4] = 0
    return indices, types, weights, basis, upstream, spikes


def _oracle(fixture, direct, scale):
    """FP64 edge equations, independent of CSR, pairs, warps and reductions."""
    indices, types, weights, basis, upstream, spikes = fixture
    ds = np.zeros(spikes.shape, np.float64)
    dw = np.zeros(weights.shape, np.float64)
    currents = np.zeros(upstream.shape, np.float64)
    ds_abs = np.zeros_like(ds)
    dw_abs = np.zeros_like(dw)
    for edge, (post, pre) in enumerate(indices):
        products = upstream[:, post].astype(np.float64) * basis[types[edge]]
        projection = products.sum(axis=1)
        source = spikes[:, pre].astype(np.float64)
        # Preserve the existing direct-CSR positivity gate and the canonical
        # pair kernel's ungated spike multiplier (including signed API inputs).
        weight_source = np.maximum(source, 0) if direct else source
        ds[:, pre] += projection * weights[edge]
        dw[edge] = np.dot(projection, weight_source)
        currents[:, post] += (
            np.maximum(source, 0)[:, None] * weights[edge] * basis[types[edge]]
        )
        ds_abs[:, pre] += abs(products).sum(axis=1) * abs(weights[edge])
        dw_abs[edge] = np.dot(abs(products).sum(axis=1), abs(weight_source))
    degrees = np.bincount(indices[:, 1], minlength=spikes.shape[1])
    # Bound FP32 rounding from four-term projection, per-lane edge sums,
    # two slot shuffles, two warps and scaling. No near-zero relative division.
    eps = np.finfo(np.float32).eps
    operations = 2 * (4 + ((degrees + 63) // 64) * 8 + 5)
    ds_bound = operations * eps / (1 - operations * eps) * ds_abs * abs(scale)
    dw_bound = 24 * eps / (1 - 24 * eps) * dw_abs
    return currents, ds * scale, dw, ds_bound, dw_bound


def _packed_cpu(fixture, direct, scale):
    """Simulate the float4 lane layout and butterfly, not a dense matmul."""
    indices, types, weights, basis, upstream, spikes = fixture
    order = np.argsort(indices[:, 1], kind="stable")
    pairs, pair_ids = np.unique(
        np.stack([indices[order, 0], types[order]], axis=1),
        axis=0, return_inverse=True,
    )
    projected = np.zeros((len(pairs), 32), np.float32)
    for receptor in range(4):
        projected += (
            upstream[:, pairs[:, 0], receptor].T
            * basis[pairs[:, 1], receptor, None]
        )
    ds = np.zeros_like(spikes)
    dw = np.empty_like(weights)
    lane = np.arange(32)
    slot, sub = lane // 8, lane % 8
    samples = sub[:, None] * 4 + np.arange(4)
    target = ((sub & 1) << 2) | (sub & 2) | ((sub & 4) >> 2)
    rows = np.r_[0, np.cumsum(np.bincount(
        indices[:, 1], minlength=spikes.shape[1]))]
    for pre in range(spikes.shape[1]):
        source = spikes[samples, pre]
        if direct:
            source = np.maximum(source, 0)
        warp_totals = []
        for warp in range(2):
            pre_grad = np.zeros((32, 4), np.float32)
            for base in range(rows[pre] + warp * 32, rows[pre + 1], 64):
                partial = np.zeros((32, 8), np.float32)
                for step in range(8):
                    edges = base + 4 * step + slot
                    valid = edges < rows[pre + 1]
                    values = np.zeros((32, 4), np.float32)
                    w = np.zeros(32, np.float32)
                    values[valid] = projected[pair_ids[edges[valid], None], samples[valid]]
                    w[valid] = weights[order[edges[valid]]]
                    for sample in range(4):
                        pre_grad[:, sample] += values[:, sample] * w
                        partial[:, step] += values[:, sample] * source[:, sample]
                if direct and not np.any(source > 0):
                    edges = np.arange(base, min(base + 32, rows[pre + 1]))
                    dw[order[edges]] = 0
                    continue
                for half, mask in [(4, 1), (2, 2), (1, 4)]:
                    upper = ((lane & mask) != 0)[:, None]
                    keep = np.where(upper, partial[:, half:2 * half], partial[:, :half])
                    send = np.where(upper, partial[:, :half], partial[:, half:2 * half])
                    partial = keep + send[lane ^ mask]
                edges = base + 4 * target + slot
                valid = edges < rows[pre + 1]
                dw[order[edges[valid]]] = partial[valid, 0]
            for mask in (8, 16):
                pre_grad += pre_grad[lane ^ mask]
            warp_totals.append(pre_grad[:8].reshape(32))
        ds[:, pre] = (warp_totals[0] + warp_totals[1]) * np.float32(scale)
    return ds, dw


def _assert_vjp(actual, expected, bound):
    error = np.abs(np.asarray(actual, np.float64) - expected)
    assert np.all(error <= bound), (
        "FP32 rounding bound exceeded: max error {}, max bound {}".format(
            error.max(), bound.max()
        )
    )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("kind", ["binary", "counts", "fractional", "signed"])
@pytest.mark.parametrize("tiny", [False, True])
def test_fp32_batch32_packed_cpu_oracle(direct, kind, tiny):
    fixture = _fixture(kind, tiny)
    scale = -0.375
    _, expected_ds, expected_dw, ds_bound, dw_bound = _oracle(fixture, direct, scale)
    ds, dw = _packed_cpu(fixture, direct, scale)
    _assert_vjp(ds, expected_ds, ds_bound)
    _assert_vjp(dw, expected_dw, dw_bound)
    np.testing.assert_array_equal(ds[:, [0, 11]], 0)
    np.testing.assert_array_equal(dw[fixture[0][:, 1] == 4], 0)
    assert np.any(ds[:, 4] != 0), "Silent rows must retain temporal credit."
    if tiny:
        assert np.any((abs(ds) > 0) & (abs(ds) < 2 ** -24))


def test_fp32_batch32_oracle_finite_difference():
    fixture = list(_fixture("fractional"))
    indices, types, weights, basis, upstream, spikes = fixture
    fixture[2] = weights.astype(np.float64)
    fixture[5] = spikes.astype(np.float64)
    _, ds, dw, _, _ = _oracle(fixture, True, 1.0)

    def objective():
        source = fixture[5][:, indices[:, 1]]
        projection = np.einsum(
            "bek,ek->be", upstream[:, indices[:, 0]].astype(np.float64),
            basis[types].astype(np.float64),
        )
        return np.sum(source * projection * fixture[2])

    for array, index, expected in [
        (fixture[2], 17, dw[17]),
        (fixture[5], (7, 10), ds[7, 10]),
    ]:
        original = array[index]
        delta = 1e-5
        array[index] = original + delta
        plus = objective()
        array[index] = original - delta
        minus = objective()
        array[index] = original
        np.testing.assert_allclose((plus - minus) / (2 * delta), expected,
                                   rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("kind", ["binary", "counts", "fractional", "signed"])
@pytest.mark.parametrize("tiny", [False, True])
def test_fp32_batch32_cuda_independent_vjp(direct, kind, tiny):
    tf = pytest.importorskip("tensorflow")
    from bmtk.simulator.dpointnet.custom_ops import (
        build_csr_connectivity, fused_cuda_available, fused_spike_currents,
        reorder_csr_values, restore_csr_values,
    )
    if not fused_cuda_available():
        pytest.skip("Requires rebuilt CUDA operators and an authorized GPU")
    fixture = _fixture(kind, tiny)
    indices, types, weights, basis, upstream, spikes = fixture
    scale = -0.375
    expected, expected_ds, expected_dw, ds_bound, dw_bound = _oracle(fixture, direct, scale)
    connectivity = build_csr_connectivity(
        indices, types, spikes.shape[1], upstream.shape[1], len(basis),
        build_compact_pairs=True,
    )
    try:
        with tf.device("/GPU:0"):
            source = tf.Variable(spikes)
            master = tf.Variable(weights)
            csr = reorder_csr_values(master, connectivity)
            with tf.GradientTape() as tape:
                currents = fused_spike_currents(
                    source, master, csr, connectivity, tf.constant(basis),
                    n_post=upstream.shape[1], compute_spike_gradient=True,
                    spike_gradient_scale=scale, use_packed_sm120_backward=False,
                    write_csr_weight_gradient=direct,
                )
                loss = tf.reduce_sum(currents * upstream.reshape(-1, 4))
            ds, dw = tape.gradient(loss, [source, master])
            if direct:
                dw = restore_csr_values(dw, connectivity)
        np.testing.assert_allclose(currents.numpy(), expected.reshape(-1, 4),
                                   rtol=2e-5, atol=2e-5)
        _assert_vjp(ds.numpy(), expected_ds, ds_bound)
        _assert_vjp(dw.numpy(), expected_dw, dw_bound)
        assert ds.dtype == tf.float32
        assert dw.dtype == tf.float32
    finally:
        connectivity.close()

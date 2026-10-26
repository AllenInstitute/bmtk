import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.custom_ops import lgn_grating_ops
from bmtk.simulator.dpointnet.input_modules.lgn_generator import DriftingGratingLGN
from bmtk.simulator.dpointnet.lgn_tf.lgn import LGN

LGN_CSV = os.environ.get(
    "DPOINTNET_LGN_CSV",
    "/local2/results/dpointnet_rule_search/javier_runtime_20260929/data/tf_data/lgn_full_col_cells_120x80.csv",
)


def _native_available():
    return lgn_grating_ops.native_grating_lgn_available()


@pytest.fixture(scope="module")
def small_lgn(tmp_path_factory):
    """A 300-unit LGN drawn from every unit type of the full 120x80 population."""
    if not os.path.exists(LGN_CSV):
        pytest.skip("Full LGN population CSV is unavailable.")
    frame = pd.read_csv(LGN_CSV, sep=" ")
    rng = np.random.default_rng(5)
    rows = np.concatenate([
        rng.choice(np.flatnonzero(frame["model_id"] == name), size=min(20, count), replace=False)
        for name, count in frame["model_id"].value_counts().items()
    ])
    subset = frame.iloc[np.sort(rows)].reset_index(drop=True)
    cache = tmp_path_factory.mktemp("lgn")
    network = SimpleNamespace(get_nodes_df=lambda: subset, n_nodes=len(subset), name="lgn")
    return LGN(
        network=network, row_size=80, col_size=120,
        spon_frs_path=str(cache / "spon.pkl"), temp_krns_path=str(cache / "temp.pkl"),
        spatial_krns_path=str(cache / "spatial.pkl"),
    )


def _generators(lgn, **options):
    native = DriftingGratingLGN(lgn, 120, pre_delay=20, post_delay=10, lgn_backend="native", **options)
    reference = DriftingGratingLGN(lgn, 120, pre_delay=20, post_delay=10, lgn_backend="tensorflow", **options)
    assert native.backend == "native" and reference.backend == "tensorflow"
    return native, reference


def test_backend_selection_and_errors(small_lgn):
    with pytest.raises(ValueError, match="lgn_backend"):
        DriftingGratingLGN(small_lgn, 120, lgn_backend="cuda")
    reference = DriftingGratingLGN(small_lgn, 120, lgn_backend="tensorflow")
    assert reference.backend == "tensorflow" and reference.kernel is None
    automatic = DriftingGratingLGN(small_lgn, 120)
    assert automatic.backend == ("native" if _native_available() else "tensorflow")
    if not _native_available():
        with pytest.raises(RuntimeError):
            DriftingGratingLGN(small_lgn, 120, lgn_backend="native")
        return
    with pytest.raises(ValueError, match="chunk"):
        lgn_grating_ops.GratingLGNKernel(
            small_lgn, seq_len=120, pre_delay=0, post_delay=0, temporal_f=2, cpd=0.04, contrast=0.8,
            rows=80, cols=120, theta_sign=1.0, theta_offset=0.0, chunk=0,
        )
    with pytest.raises(ValueError, match="delays"):
        DriftingGratingLGN(small_lgn, 120, pre_delay=100, post_delay=100, lgn_backend="native")


@pytest.mark.parametrize("rotation,billeh_phase", [("ccw", False), ("cw", True)])
def test_native_rates_match_tensorflow_path(small_lgn, rotation, billeh_phase):
    """The exact factorization reproduces the movie -> spatial -> temporal rates
    to float32 rounding for every unit type, both rotations and the phase offset."""
    if not _native_available():
        pytest.skip("Native LGN grating operators and a compatible GPU are required.")
    native, reference = _generators(small_lgn, rotation=rotation, billeh_phase=billeh_phase)
    for theta, phase in ((0.0, 0.0), (37.5, 123.0), (271.0, 359.0)):
        expected = reference.firing_rates(tf.constant(theta), tf.constant(phase)).numpy()
        actual = native.firing_rates(tf.constant(theta), tf.constant(phase)).numpy()
        assert actual.shape == expected.shape == (120, small_lgn.n_nodes)
        assert np.all(actual >= 0) and actual.mean() > 0.1
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-3)
        # Gray screen before and after the grating: both paths agree there too.
        np.testing.assert_allclose(actual[:20], expected[:20], rtol=1e-5, atol=1e-4)


def test_native_batch_spikes_match_tensorflow_path_and_are_seeded(small_lgn):
    if not _native_available():
        pytest.skip("Native LGN grating operators and a compatible GPU are required.")
    native, reference = _generators(small_lgn)
    rng = np.random.default_rng(11)
    batch = 5
    theta = tf.constant(rng.uniform(0, 360, (batch, 1)).astype(np.float32))
    phase = tf.constant(rng.uniform(0, 360, batch).astype(np.float32))
    seeds = tf.constant(rng.integers(0, 2**31 - 1, (batch, 2)).astype(np.int32))
    actual = native.batch_spikes(theta, phase, seeds).numpy()
    expected = reference.batch_spikes(theta, phase, seeds).numpy()
    assert actual.shape == expected.shape == (batch, 120, small_lgn.n_nodes) and actual.dtype == np.bool_
    # Same stateless uniforms; probabilities differ only by float32 rounding.
    mismatch = np.mean(actual != expected)
    assert mismatch < 1e-4, mismatch
    assert abs(actual.mean() - expected.mean()) / expected.mean() < 1e-2
    np.testing.assert_array_equal(native.batch_spikes(theta, phase, seeds).numpy(), actual)
    # Chunking over the batch does not change the samples.
    chunked = DriftingGratingLGN(small_lgn, 120, pre_delay=20, post_delay=10, lgn_backend="native", native_chunk=2)
    np.testing.assert_array_equal(chunked.batch_spikes(theta, phase, seeds).numpy(), actual)
    single = native.spikes(theta[1, 0], phase[1], seeds[1]).numpy()
    np.testing.assert_array_equal(single, actual[1])


def test_native_current_input_matches_tensorflow_path(small_lgn):
    if not _native_available():
        pytest.skip("Native LGN grating operators and a compatible GPU are required.")
    native, reference = _generators(small_lgn, current_input=True)
    theta = tf.constant([[12.0], [200.0]]); phase = tf.constant([5.0, 90.0])
    seeds = tf.zeros((2, 2), tf.int32)
    np.testing.assert_allclose(
        native.batch_spikes(theta, phase, seeds).numpy(), reference.batch_spikes(theta, phase, seeds).numpy(),
        rtol=1e-4, atol=1e-5,
    )


def test_native_ops_run_inside_tf_function(small_lgn):
    if not _native_available():
        pytest.skip("Native LGN grating operators and a compatible GPU are required.")
    native, _ = _generators(small_lgn)
    theta = tf.constant([[45.0], [90.0], [135.0]]); phase = tf.constant([0.0, 10.0, 20.0])
    seeds = tf.constant([[1, 2], [3, 4], [5, 6]], tf.int32)
    eager = native.batch_spikes(theta, phase, seeds).numpy()
    compiled = tf.function(native.batch_spikes)(theta, phase, seeds).numpy()
    np.testing.assert_array_equal(compiled, eager)


def test_native_symbolic_batch_matches_static_batch(small_lgn):
    """A tf.function with an unknown batch dimension samples one trial at a time
    and reproduces the statically batched spikes."""
    if not _native_available():
        pytest.skip("Native LGN grating operators and a compatible GPU are required.")
    native, _ = _generators(small_lgn)
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, 360, (3, 1)).astype(np.float32)
    phase = rng.uniform(0, 360, 3).astype(np.float32)
    seeds = rng.integers(0, 2**31 - 1, (3, 2)).astype(np.int32)
    static = native.batch_spikes(tf.constant(theta), tf.constant(phase), tf.constant(seeds)).numpy()
    symbolic = tf.function(native.batch_spikes, input_signature=(
        tf.TensorSpec((None, 1), tf.float32), tf.TensorSpec((None,), tf.float32), tf.TensorSpec((None, 2), tf.int32),
    ))
    np.testing.assert_array_equal(symbolic(theta, phase, seeds).numpy(), static)
    np.testing.assert_array_equal(symbolic(theta[:2], phase[:2], seeds[:2]).numpy(), static[:2])


def test_unsupported_dtype_keeps_tensorflow_under_auto(small_lgn):
    automatic = DriftingGratingLGN(small_lgn, 120, dtype=tf.float64)
    assert automatic.backend == "tensorflow"
    rates = automatic.firing_rates(tf.constant(30.0, tf.float64), tf.constant(0.0, tf.float64))
    assert rates.shape == (120, small_lgn.n_nodes) and np.all(np.isfinite(rates.numpy()))
    with pytest.raises(ValueError, match="float16 or float32"):
        DriftingGratingLGN(small_lgn, 120, dtype=tf.float64, lgn_backend="native")


def test_host_generator_accepts_device_backend_options(small_lgn):
    """The host reference generator (initial-state warmup) tolerates the device
    route's stimulus options so one option dictionary serves both routes."""
    from bmtk.simulator.dpointnet.input_modules.lgn_generator import create_drifting_gratings_generator

    dataset = create_drifting_gratings_generator(
        small_lgn, 40, orientation=[90.0], phase=10.0, pre_delay=5, post_delay=5, seed=3,
        lgn_backend="tensorflow", native_chunk=4,
    )
    spikes, meta = next(iter(dataset))
    assert spikes.shape == (40, small_lgn.n_nodes) and spikes.dtype == tf.bool
    assert float(meta["orientation"][0]) == 90.0

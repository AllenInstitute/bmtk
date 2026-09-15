import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.lgn_tf.lgn import (
    LGN,
    _bilinear_metadata,
    _prepare_spatial_filters,
)
from bmtk.simulator.dpointnet.input_modules.lgn_generator import (
    _tensorflow_uniform_scalar,
    create_drifting_gratings_generator,
    make_drifting_grating_stimulus,
)


def test_drifting_grating_accepts_repeated_scalar_tensor_parameters():
    movies = [
        make_drifting_grating_stimulus(
            row_size=5,
            col_size=7,
            image_duration=4,
            theta=tf.constant(float(theta)),
            phase=tf.constant(float(theta + 15)),
        )
        for theta in range(0, 360, 15)
    ]

    assert all(movie.shape == (4, 5, 7) for movie in movies)
    assert all(np.isfinite(movie.numpy()).all() for movie in movies)
    assert not np.array_equal(movies[0].numpy(), movies[-1].numpy())


def test_drifting_grating_dataset_repeats_without_scalar_aliasing():
    class FakeLGN:
        n_nodes = 4

        @staticmethod
        def spatial_response(videos, bmtk_compat):
            length = tf.shape(videos)[0]
            return tf.zeros((length, 4)), tf.zeros((length, 0))

        @staticmethod
        def firing_rates_from_spatial(dominant, non_dominant):
            return tf.ones_like(dominant) * 5.0

    dataset = (
        create_drifting_gratings_generator(
            FakeLGN(),
            seq_len=8,
            row_size=5,
            col_size=7,
            seed=3000,
        )
        .batch(5)
        .prefetch(tf.data.AUTOTUNE)
    )

    batches = list(dataset.take(20))

    assert len(batches) == 20
    assert all(spikes.shape == (5, 8, 4) for spikes, _ in batches)
    orientations = np.concatenate(
        [targets["orientation"].numpy() for _, targets in batches]
    )
    assert orientations.shape == (100, 1)
    assert np.isfinite(orientations).all()


def test_tensorflow_uniform_scalar_is_repeatable_python_value():
    seed = tf.constant([3000, 1001], dtype=tf.int32)

    first = _tensorflow_uniform_scalar(0, 360, tf.float32, seed=seed)
    second = _tensorflow_uniform_scalar(0, 360, tf.float32, seed=seed)

    assert isinstance(first, float)
    assert first == second


def test_fused_separable_spatial_response_matches_full_convolutions():
    row_size = 7
    col_size = 8
    vertical = [
        np.array([1.0, 2.0, 1.0], dtype=np.float32),
        np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32),
    ]
    horizontal = [
        np.array([1.0, 3.0, 1.0], dtype=np.float32),
        np.array([1.0, 2.0, 1.0], dtype=np.float32),
    ]
    gaussian_filters = [
        np.outer(v, h)[..., None, None] / np.outer(v, h).sum()
        for v, h in zip(vertical, horizontal)
    ]
    packed_vertical, packed_horizontal, edge_reciprocals = _prepare_spatial_filters(
        gaussian_filters, row_size, col_size, tf.float32
    )

    x = np.array([1.25, 5.4, 3.5], dtype=np.float32)
    y = np.array([2.5, 4.25, 1.75], dtype=np.float32)
    non_dominant_x = np.array([0.0, 4.7, 2.2], dtype=np.float32)
    non_dominant_y = np.array([0.0, 3.1, 5.2], dtype=np.float32)
    groups = [np.array([0, 2], dtype=np.int32), np.array([1], dtype=np.int32)]
    composite_mask = np.array([False, True, True])

    lgn = object.__new__(LGN)
    lgn.dtype = tf.float32
    lgn.packed_vertical_filters = packed_vertical
    lgn.packed_horizontal_filters = packed_horizontal
    lgn.edge_reciprocals = edge_reciprocals
    lgn.spatial_range_indices = groups
    lgn.sorted_neuron_ids_indices = tf.constant([0, 2, 1], dtype=tf.int32)
    lgn.dominant_sample_indices = []
    lgn.dominant_sample_weights = []
    lgn.non_dominant_sample_indices = []
    lgn.non_dominant_sample_weights = []
    grouped_composite_ids = []
    for indices in groups:
        dominant_indices, dominant_weights = _bilinear_metadata(
            x[indices], y[indices], col_size
        )
        selected = indices[composite_mask[indices]]
        non_dominant_indices, non_dominant_weights = _bilinear_metadata(
            non_dominant_x[selected], non_dominant_y[selected], col_size
        )
        lgn.dominant_sample_indices.append(tf.constant(dominant_indices))
        lgn.dominant_sample_weights.append(tf.constant(dominant_weights))
        lgn.non_dominant_sample_indices.append(tf.constant(non_dominant_indices))
        lgn.non_dominant_sample_weights.append(tf.constant(non_dominant_weights))
        grouped_composite_ids.append(selected)
    lgn.composite_sort_indices = tf.constant(
        np.argsort(np.concatenate(grouped_composite_ids)).astype(np.int32)
    )

    movie = tf.random.stateless_uniform(
        (4, row_size, col_size, 1), seed=(11, 29), dtype=tf.float32
    )
    actual_dominant, actual_non_dominant = lgn.spatial_response(movie, bmtk_compat=True)

    reference_dominant = np.empty((4, 3), dtype=np.float32)
    reference_non_dominant = np.empty((4, 2), dtype=np.float32)
    for kernel, indices, reciprocal in zip(gaussian_filters, groups, edge_reciprocals):
        convolved = tf.nn.conv2d(movie, kernel, strides=1, padding="SAME")
        convolved = (convolved * reciprocal)[..., 0]
        dominant = LGN.select_spatial(
            tf.constant(x[indices]), tf.constant(y[indices]), convolved
        ).numpy()
        reference_dominant[:, indices] = dominant
        selected = indices[composite_mask[indices]]
        non_dominant = LGN.select_spatial(
            tf.constant(non_dominant_x[selected]),
            tf.constant(non_dominant_y[selected]),
            convolved,
        ).numpy()
        for column, neuron_id in enumerate(selected):
            reference_non_dominant[
                :, np.flatnonzero(np.flatnonzero(composite_mask) == neuron_id)[0]
            ] = non_dominant[:, column]

    np.testing.assert_allclose(actual_dominant, reference_dominant, atol=1e-6)
    np.testing.assert_allclose(actual_non_dominant, reference_non_dominant, atol=1e-6)


def test_compact_composite_firing_rates_match_full_population_path():
    lgn = object.__new__(LGN)
    lgn.dtype = tf.float32
    lgn.dom_temporal_kernels = tf.constant(
        [[0.5, 0.25, 0.75], [0.5, 0.75, 0.25]], dtype=tf.float32
    )
    full_non_dom_kernels = tf.constant(
        [[0.0, 0.4, 0.0], [0.0, 0.6, 0.0]], dtype=tf.float32
    )
    lgn.composite_ids = tf.constant([1], dtype=tf.int32)
    lgn.n_composite = 1
    lgn.composite_non_dom_kernels = tf.gather(
        full_non_dom_kernels, lgn.composite_ids, axis=1
    )
    lgn.amplitude = tf.constant([1.0, -1.0, 0.5], dtype=tf.float32)
    lgn.non_dom_amplitude = tf.constant([0.0, 0.8, 0.0], dtype=tf.float32)
    lgn.composite_non_dom_amplitude = tf.gather(
        lgn.non_dom_amplitude, lgn.composite_ids
    )
    lgn.spontaneous_firing_rates = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)
    lgn.composite_spontaneous_rates = tf.gather(
        lgn.spontaneous_firing_rates, lgn.composite_ids
    )
    dominant = tf.random.stateless_uniform((6, 3), seed=(5, 7))
    compact_non_dominant = tf.random.stateless_uniform((6, 1), seed=(11, 13))

    actual = lgn.firing_rates_from_spatial(dominant, compact_non_dominant)

    full_non_dominant = tf.scatter_nd(
        [[1]],
        tf.transpose(compact_non_dominant),
        (3, 6),
    )
    full_non_dominant = tf.transpose(full_non_dominant)
    reference_dominant = LGN.transfer_function(
        LGN.temporal_filter(dominant, lgn.dom_temporal_kernels) * lgn.amplitude
        + lgn.spontaneous_firing_rates
    )
    reference_non_dominant = LGN.transfer_function(
        LGN.temporal_filter(full_non_dominant, full_non_dom_kernels)
        * lgn.non_dom_amplitude
        + lgn.spontaneous_firing_rates
    )
    reference = (
        reference_dominant + tf.constant([0.0, 1.0, 0.0]) * reference_non_dominant
    )

    np.testing.assert_allclose(actual, reference, atol=1e-6)

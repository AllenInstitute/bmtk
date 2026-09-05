import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle as pkl
import logging

from bmtk.simulator.dpointnet.io_tools import io
from bmtk.simulator.filternet.lgnmodel.fitfuns import makeBasis_StimKernel
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.util_fns import get_tcross_from_temporal_kernel
from bmtk.simulator.filternet.lgnmodel.cellmetrics import get_data_metrics_for_each_subclass

try:
    from numba import njit

    @njit(cache=True)
    def _assign_spatial_bin_ids(spatial_sizes, spatial_range):
        """Assign each spatial size to a [low, high) range bin index, or -1 if none."""
        n = spatial_sizes.shape[0]
        n_bins = spatial_range.shape[0] - 1
        out = np.full(n, -1, dtype=np.int32)
        for idx in range(n):
            v = spatial_sizes[idx]
            for b in range(n_bins):
                if (v >= spatial_range[b]) and (v < spatial_range[b + 1]):
                    out[idx] = b
                    break
        return out

except Exception as e:
    def _assign_spatial_bin_ids(spatial_sizes, spatial_range):
        n_bins = len(spatial_range) - 1
        out = np.full(spatial_sizes.shape[0], -1, dtype=np.int32)
        for b in range(n_bins):
            sel = np.logical_and(spatial_sizes >= spatial_range[b], spatial_sizes < spatial_range[b + 1])
            out[sel] = b
        return out


logger = logging.getLogger(__name__)


def _bilinear_metadata(x, y, width):
    """Precompute flattened indices and weights for bilinear sampling."""
    x0 = np.floor(x).astype(np.int32)
    x1 = np.ceil(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.ceil(y).astype(np.int32)
    x_fraction = x - x0
    y_fraction = y - y0
    indices = np.stack(
        (y0 * width + x0, y1 * width + x0, y0 * width + x1, y1 * width + x1),
        axis=1,
    )
    weights = np.stack(
        (
            (1 - x_fraction) * (1 - y_fraction),
            (1 - x_fraction) * y_fraction,
            x_fraction * (1 - y_fraction),
            x_fraction * y_fraction,
        ),
        axis=1,
    ).astype(np.float32)
    return indices, weights


def _sample_spatial(flattened_movie, indices, weights):
    values = tf.gather(flattened_movie, indices, axis=1)
    return tf.reduce_sum(values * weights[None, ...], axis=-1)


def _prepare_spatial_filters(gaussian_filters, row_size, col_size, dtype):
    vertical_filters = []
    horizontal_filters = []
    edge_reciprocals = []
    for gaussian_filter in gaussian_filters:
        matrix = np.asarray(gaussian_filter)[:, :, 0, 0]
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if singular_values[1:].sum() > singular_values[0] * 1e-5:
            raise ValueError("Expected a rank-one Gaussian spatial kernel.")
        scale = np.sqrt(singular_values[0])
        vertical_filters.append((left[:, 0] * scale)[:, None, None, None])
        horizontal_filters.append((right[0] * scale)[None, :, None, None])
        edge_fraction = tf.nn.conv2d(
            tf.ones((1, row_size, col_size, 1), dtype=dtype),
            tf.constant(gaussian_filter, dtype=dtype),
            strides=1,
            padding="SAME",
        )
        edge_reciprocals.append(tf.math.reciprocal(edge_fraction))

    max_vertical = max(value.shape[0] for value in vertical_filters)
    max_horizontal = max(value.shape[1] for value in horizontal_filters)

    def center_pad(value, target, axis):
        padding = target - value.shape[axis]
        before = padding // 2
        after = padding - before
        widths = [(0, 0)] * value.ndim
        widths[axis] = (before, after)
        return np.pad(value, widths)

    packed_vertical = np.concatenate(
        [center_pad(value, max_vertical, axis=0) for value in vertical_filters], axis=3
    )
    packed_horizontal = np.concatenate(
        [center_pad(value, max_horizontal, axis=1) for value in horizontal_filters],
        axis=2,
    )
    return (
        tf.constant(packed_vertical, dtype=dtype),
        tf.constant(packed_horizontal, dtype=dtype),
        edge_reciprocals,
    )


class LGN:
    def __init__(
        self,
        network,
        row_size,
        col_size,
        dtype=tf.float32,
        spon_frs_path=None,
        temp_krns_path=None,
        spatial_krns_path=None,
    ):
        # Load table of LGN nodes
        # node_types_df = pd.read_csv(node_types_file, sep=' ')
        # if 'population' in node_types_df:
        #     node_types_df = node_types_df[node_types_df['population'] == population]

        # with h5py.File(nodes_file, 'r') as h5:
        #     nodes_grp = h5['/nodes'][population]
        #     nodes_df = pd.DataFrame({
        #         'node_id': nodes_grp['node_id'],
        #         'node_type_id': nodes_grp['node_type_id'],
        #         'node_group_id': nodes_grp['node_group_id'],
        #         'node_group_index': nodes_grp['node_group_index'],
        #     })
        #     node_grp_ids = np.unique(nodes_grp['node_group_id'][()])
        #     for grp_id in node_grp_ids:
        #         model_grp = nodes_grp[str(grp_id)]
        #         model_df = pd.DataFrame({n: v for n, v in model_grp.items() if isinstance(v, h5py.Dataset)})
        #         model_df['node_group_id']= grp_id
        #         model_df['node_group_index'] = np.arange(len(model_df))
        #         nodes_df = pd.merge(nodes_df, model_df, how='left', on=['node_group_id', 'node_group_index'])
        #         nodes_df = nodes_df.drop(columns=['node_group_id', 'node_group_index'])

        #     # If there are any columns in both h5 and csv, remove them from node_types_df
        #     shared_attrs = (set(nodes_df.columns) & set(node_types_df.columns))  - set(['node_type_id'])
        #     if shared_attrs:
        #         node_types_df = node_types_df.drop(columns=shared_attrs)

        #     nodes_df = pd.merge(nodes_df, node_types_df, how='left', on='node_type_id')
        #     if 'pop_name' in nodes_df and 'model_id' not in nodes_df:
        #         nodes_df = nodes_df.rename(columns={'pop_name': 'model_id'})

        nodes_df = network.get_nodes_df()
        if 'pop_name' in nodes_df and 'model_id' not in nodes_df:
            nodes_df = nodes_df.rename(columns={'pop_name': 'model_id'})

        # n_units = len(nodes_df)
        self.dtype = dtype
        self.n_nodes = len(nodes_df)

        model_ids = nodes_df['model_id'].to_numpy()
        amplitudes = np.array([1.0 if m.count('ON') > 0 else -1.0 for m in model_ids], dtype=np.float32)
        non_dom_amplitudes = np.zeros_like(amplitudes)
        is_composite = np.array([('ON' in m and 'OFF' in m) for m in model_ids], dtype=np.float32)

        # Load the spontaneous firing rates
        if spon_frs_path is None or not Path(spon_frs_path).exists():
            cell_types = [m[:m.find('_')] for m in model_ids]
            freqs = [a[a.find('_') + 1:] for a in model_ids]
            spontaneous_firing_rates = []

            for t, f in zip(cell_types, freqs):
                if t.count('ON') > 0 and t.count('OFF') > 0:
                    spontaneous_firing_rates.append(-1.0)
                else:
                    spontaneous_firing_rate = get_data_metrics_for_each_subclass(t)[f]['spont_exp']
                    spontaneous_firing_rates.append(spontaneous_firing_rate[0])
            spontaneous_firing_rates = np.array(spontaneous_firing_rates, dtype=np.float32)

            if spon_frs_path is not None:
                with open(spon_frs_path, 'wb') as f:
                    pkl.dump(spontaneous_firing_rates, f)
        else:
            with open(spon_frs_path, 'rb') as f:
                spontaneous_firing_rates = np.asarray(pkl.load(f), dtype=np.float32)

        # Load the temporal kernels
        if temp_krns_path is None or not Path(temp_krns_path).exists():
            nkt = 600
            kernel_length = 700
            dom_temporal_kernels = []
            non_dom_temporal_kernels = []
            logger.debug('Computing temporal kernels')
            # Load spatial features of the elliptical subfields
            tuning_angle = nodes_df['tuning_angle'].to_numpy(dtype=np.float32)
            subfield_separation = nodes_df['sf_sep'].to_numpy(dtype=np.float32)
            x = nodes_df['x'].to_numpy(dtype=np.float32)
            y = nodes_df['y'].to_numpy(dtype=np.float32)
            non_dominant_x = np.zeros_like(x)
            non_dominant_y = np.zeros_like(y)

            # Load the temporal kernels features
            temporal_peaks_dom = np.stack(
                (nodes_df['kpeaks_dom_0'].to_numpy(dtype=np.float32), nodes_df['kpeaks_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_weights = np.stack(
                (nodes_df['weight_dom_0'].to_numpy(dtype=np.float32), nodes_df['weight_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_delays = np.stack(
                (nodes_df['delay_dom_0'].to_numpy(dtype=np.float32), nodes_df['delay_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )

            temporal_peaks_non_dom = np.stack(
                (nodes_df['kpeaks_non_dom_0'].to_numpy(dtype=np.float32), nodes_df['kpeaks_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_weights_non_dom = np.stack(
                (nodes_df['weight_non_dom_0'].to_numpy(dtype=np.float32), nodes_df['weight_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )
            temporal_delays_non_dom = np.stack(
                (nodes_df['delay_non_dom_0'].to_numpy(dtype=np.float32), nodes_df['delay_non_dom_1'].to_numpy(dtype=np.float32)),
                axis=-1,
            )

            for i in range(x.shape[0]):
                dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                non_dom_temporal_kernel = np.zeros((kernel_length,), np.float32)
                if model_ids[i].count('ON') > 0 and model_ids[i].count('OFF') > 0:
                    non_dom_params = dict(
                        opt_wts=temporal_weights_non_dom[i],
                        opt_kpeaks=temporal_peaks_non_dom[i],
                        opt_delays=temporal_delays_non_dom[i],
                    )
                    dom_params = dict(
                        opt_wts=temporal_weights[i],
                        opt_kpeaks=temporal_peaks_dom[i],
                        opt_delays=temporal_delays[i],
                    )
                    amp_on = 1.0 # set the non-dominant subunit amplitude to unity

                    if model_ids[i].count('sONsOFF_001') > 0:
                        non_dom_filter, non_dom_sum = LGN.create_one_unit_of_two_subunit_filter(non_dom_params, 121.0)
                        dom_filter, dom_sum = LGN.create_one_unit_of_two_subunit_filter(dom_params, 115.0)
                        spont = 4.0
                        max_roff = 35.0
                        max_ron = 21.0
                        amp_off = -(max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    elif model_ids[i].count('sONtOFF_001') > 0:
                        non_dom_filter, non_dom_sum = LGN.create_one_unit_of_two_subunit_filter(non_dom_params, 93.5)
                        dom_filter, dom_sum = LGN.create_one_unit_of_two_subunit_filter(dom_params, 64.8)
                        spont = 5.5
                        max_roff = 46.0
                        max_ron = 31.0
                        amp_off = -0.7 * (max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on - (
                            spont * (max_roff - max_ron)) / (max_ron * dom_sum)
                    else:
                        raise ValueError('Unknown cell type')

                    non_dom_amplitudes[i] = amp_on
                    amplitudes[i] = amp_off
                    spontaneous_firing_rates[i] = spont / 2.0

                    hor_offset = np.cos(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + x[i]
                    vert_offset = np.sin(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + y[i]
                    non_dominant_x[i] = hor_offset
                    non_dominant_y[i] = vert_offset
                    dom_temporal_kernel[-len(dom_filter.kernel_data):] = dom_filter.kernel_data[::-1]
                    non_dom_temporal_kernel[-len(non_dom_filter.kernel_data):] = non_dom_filter.kernel_data[::-1]
                else:
                    dd = dict(
                        neye=0,
                        ncos=2,
                        kpeaks=temporal_peaks_dom[i],
                        b=0.3,
                        delays=[temporal_delays[i].astype(int)],
                    )
                    kernel_data = np.dot(makeBasis_StimKernel(dd, nkt), temporal_weights[i])
                    dom_temporal_kernel[-len(kernel_data):] = kernel_data

                dom_temporal_kernels.append(dom_temporal_kernel)
                non_dom_temporal_kernels.append(non_dom_temporal_kernel)

            dom_temporal_kernels = np.asarray(dom_temporal_kernels, dtype=np.float32)
            non_dom_temporal_kernels = np.asarray(non_dom_temporal_kernels, dtype=np.float32)

            # Apply truncation
            dom_cumsum = np.cumsum(np.abs(dom_temporal_kernels), axis=1)
            non_dom_cumsum = np.cumsum(np.abs(non_dom_temporal_kernels), axis=1)
            # Find the minimum number of steps where cumulative sum is below threshold
            threshold = 1e-6
            # For dominant kernels: compute truncation points for every filters
            dom_truncation_points = np.sum(dom_cumsum <= threshold, axis=1)
            # For non-dominant kernels: only include filters that are non-zero in the truncation calculation
            non_dom_truncation_points = np.where(
                np.sum(np.abs(non_dom_temporal_kernels), axis=1) > 0,
                np.sum(non_dom_cumsum <= threshold, axis=1),
                np.inf,
            )
            # Find the minimum truncation point while ignoring zero filters (set to np.inf to avoid affecting the min calculation)
            dom_truncation = int(np.min(dom_truncation_points))
            # non_dom_truncation = int(np.min(non_dom_truncation_points))
            # Handle the case where all non-dominant kernels are zero (no composite cells)
            if np.all(np.isinf(non_dom_truncation_points)):
                # No composite cells, use only dominant truncation
                non_dom_truncation = dom_truncation
            else:
                non_dom_truncation = int(np.min(non_dom_truncation_points))

            # Apply the truncation to both dominant and non-dominant temporal kernels
            truncation = int(np.min([dom_truncation, non_dom_truncation]))
            # Truncate and transpose the kernels from the truncation point onwards
            dom_temporal_kernels = dom_temporal_kernels[:, dom_truncation:].T
            non_dom_temporal_kernels = non_dom_temporal_kernels[:, non_dom_truncation:].T
            logger.debug(f'Kernels truncated from time step {truncation} onwards.')

            if temp_krns_path is not None:
                to_save = dict(
                    dom_temporal_kernels=dom_temporal_kernels,
                    non_dom_temporal_kernels=non_dom_temporal_kernels,
                    non_dominant_x=non_dominant_x,
                    non_dominant_y=non_dominant_y,
                    amplitude=amplitudes.astype(np.float32),
                    non_dom_amplitude=non_dom_amplitudes.astype(np.float32),
                    spontaneous_firing_rates=np.asarray(spontaneous_firing_rates, dtype=np.float32),
                )
                with open(temp_krns_path, 'wb') as f:
                    pkl.dump(to_save, f)

        else:
            with open(temp_krns_path, 'rb') as f:
                loaded = pkl.load(f)
            dom_temporal_kernels = np.asarray(loaded['dom_temporal_kernels'], dtype=np.float32)
            non_dom_temporal_kernels = np.asarray(loaded['non_dom_temporal_kernels'], dtype=np.float32)
            non_dominant_x = np.asarray(loaded['non_dominant_x'], dtype=np.float32)
            non_dominant_y = np.asarray(loaded['non_dominant_y'], dtype=np.float32)
            amplitudes = np.asarray(loaded['amplitude'], dtype=np.float32)
            non_dom_amplitudes = np.asarray(loaded['non_dom_amplitude'], dtype=np.float32)
            spontaneous_firing_rates = np.asarray(loaded['spontaneous_firing_rates'], dtype=np.float32)
            x = nodes_df['x'].to_numpy(dtype=np.float32)
            y = nodes_df['y'].to_numpy(dtype=np.float32)

        # Load the spatial kernels
        if spatial_krns_path is None or not Path(spatial_krns_path).exists():
            logger.debug('Computing spatial kernels...')
            # Scale x and y within the range
            col_max = float(col_size - 1)
            row_max = float(row_size - 1)
            # Clamp x and y to stay within [0, col_max] and [0, row_max] respectively
            x = np.clip(x * col_max / col_size, 0, col_max)
            y = np.clip(y * row_max / row_size, 0, row_max)
            # Clamp non_dominant_x and non_dominant_y to stay within [0, col_max] and [0, row_max] respectively
            non_dominant_x = np.clip(non_dominant_x * col_max / col_size, 0, col_max)
            non_dominant_y = np.clip(non_dominant_y * row_max / row_size, 0, row_max)

            # prepare the spatial kernels in advance and store in TF format
            d_spatial = 1.0
            spatial_range = np.arange(0, 15, d_spatial, dtype=np.float32)
            x_range = np.arange(-50, 51)
            y_range = np.arange(-50, 51)
            # Load the spatial sizes of the LGN units
            spatial_sizes = nodes_df['spatial_size'].to_numpy(dtype=np.float32)

            bin_ids = _assign_spatial_bin_ids(spatial_sizes, spatial_range)
            gaussian_filters = []
            spatial_range_indices = []

            for i in range(len(spatial_range) - 1):
                # check if there is any neuron in the spatial range
                indices = np.where(bin_ids == i)[0].astype(np.int32)
                if indices.size == 0:
                    continue
                # Precompute indices for each spatial range during initialization
                spatial_range_indices.append(indices)
                # considering the spatial range as 3 x sigma of the gaussian filter, we can compute the sigma of the Gaussian filters as:
                sigma = (spatial_range[i] + d_spatial / 2.0) / 3.0
                original_filter = GaussianSpatialFilter(
                    translate=(0.0, 0.0), sigma=(sigma, sigma), origin=(0.0, 0.0)
                )
                kernel = original_filter.get_kernel(x_range, y_range, amplitude=1.0).full()
                nonzero_inds = np.where(np.abs(kernel) > 1e-9)
                rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
                cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
                kernel = kernel[rm:rM + 1, cm:cM + 1]
                gaussian_filter = kernel[..., None, None].astype(np.float32, copy=False)
                gaussian_filters.append(gaussian_filter)

            # Concatenate all the ids and sort them
            if len(spatial_range_indices) > 0:
                neuron_ids = np.concatenate(spatial_range_indices, axis=0).astype(np.int32, copy=False)
                sorted_neuron_ids_indices = np.argsort(neuron_ids).astype(np.int32, copy=False)
            else:
                sorted_neuron_ids_indices = np.empty((0,), dtype=np.int32)

            # Save the spatial kernels
            if spatial_krns_path is not None:
                to_save = dict(
                    x=x,
                    y=y,
                    non_dominant_x=non_dominant_x,
                    non_dominant_y=non_dominant_y,
                    gaussian_filters=gaussian_filters,
                    spatial_range_indices=spatial_range_indices,
                    sorted_neuron_ids_indices=sorted_neuron_ids_indices,
                )
                with open(spatial_krns_path, 'wb') as f:
                    pkl.dump(to_save, f)
                    logger.debug('Caching spatial kernels...')
        else:
            with open(spatial_krns_path, 'rb') as f:
                loaded = pkl.load(f)
            x = np.asarray(loaded['x'], dtype=np.float32)
            y = np.asarray(loaded['y'], dtype=np.float32)
            non_dominant_x = np.asarray(loaded['non_dominant_x'], dtype=np.float32)
            non_dominant_y = np.asarray(loaded['non_dominant_y'], dtype=np.float32)
            gaussian_filters = [np.asarray(gf, dtype=np.float32) for gf in loaded['gaussian_filters']]
            spatial_range_indices = [np.asarray(a, dtype=np.int32) for a in loaded['spatial_range_indices']]
            sorted_neuron_ids_indices = np.asarray(loaded['sorted_neuron_ids_indices'], dtype=np.int32)

        # Preprocess data tensors outside the loop if they don't change
        self.x = tf.constant(x, dtype=dtype)
        self.y = tf.constant(y, dtype=dtype)
        self.non_dominant_x = tf.constant(non_dominant_x, dtype=dtype)
        self.non_dominant_y = tf.constant(non_dominant_y, dtype=dtype)
        self.amplitude = tf.constant(amplitudes, dtype=dtype)
        self.non_dom_amplitude = tf.constant(non_dom_amplitudes, dtype=dtype)
        self.is_composite = tf.constant(is_composite, dtype=dtype)
        self.spontaneous_firing_rates = tf.constant(spontaneous_firing_rates, dtype=dtype)

        self.dom_temporal_kernels = tf.convert_to_tensor(
            dom_temporal_kernels, dtype=dtype
        )
        self.non_dom_temporal_kernels = tf.convert_to_tensor(
            non_dom_temporal_kernels, dtype=dtype
        )
        self.gaussian_filters = [
            tf.convert_to_tensor(gf, dtype=dtype) for gf in gaussian_filters
        ]
        self.spatial_range_indices = spatial_range_indices
        self.sorted_neuron_ids_indices = tf.convert_to_tensor(
            sorted_neuron_ids_indices, dtype=tf.int32
        )

        (
            self.packed_vertical_filters,
            self.packed_horizontal_filters,
            self.edge_reciprocals,
        ) = _prepare_spatial_filters(gaussian_filters, row_size, col_size, dtype)

        composite_mask = is_composite.astype(bool)
        composite_ids = np.flatnonzero(composite_mask).astype(np.int32)
        self.n_composite = composite_ids.size
        self.composite_ids = tf.constant(composite_ids)
        self.composite_non_dom_kernels = tf.gather(
            self.non_dom_temporal_kernels, self.composite_ids, axis=1
        )
        self.composite_non_dom_amplitude = tf.gather(
            self.non_dom_amplitude, self.composite_ids
        )
        self.composite_spontaneous_rates = tf.gather(
            self.spontaneous_firing_rates, self.composite_ids
        )

        self.dominant_sample_indices = []
        self.dominant_sample_weights = []
        self.non_dominant_sample_indices = []
        self.non_dominant_sample_weights = []
        grouped_composite_ids = []
        for indices in spatial_range_indices:
            dominant_indices, dominant_weights = _bilinear_metadata(
                x[indices], y[indices], col_size
            )
            selected_ids = indices[composite_mask[indices]]
            non_dominant_indices, non_dominant_weights = _bilinear_metadata(
                non_dominant_x[selected_ids],
                non_dominant_y[selected_ids],
                col_size,
            )
            self.dominant_sample_indices.append(tf.constant(dominant_indices))
            self.dominant_sample_weights.append(
                tf.constant(dominant_weights, dtype=dtype)
            )
            self.non_dominant_sample_indices.append(tf.constant(non_dominant_indices))
            self.non_dominant_sample_weights.append(
                tf.constant(non_dominant_weights, dtype=dtype)
            )
            grouped_composite_ids.append(selected_ids)
        grouped_composite_ids = np.concatenate(grouped_composite_ids)
        self.composite_sort_indices = tf.constant(
            np.argsort(grouped_composite_ids).astype(np.int32)
        )

    @tf.function
    def spatial_response(self, movie, bmtk_compat=True):
        """Return dominant responses and compact composite-cell responses."""
        movie = tf.cast(movie, dtype=self.dtype)
        convolved_movies = tf.nn.conv2d(
            movie, self.packed_vertical_filters, strides=1, padding="SAME"
        )
        convolved_movies = tf.nn.depthwise_conv2d(
            convolved_movies,
            self.packed_horizontal_filters,
            strides=(1, 1, 1, 1),
            padding="SAME",
        )
        all_spatial_responses = []
        all_non_dom_spatial_responses = []
        for i, _ in enumerate(self.spatial_range_indices):
            convolved_movie = convolved_movies[..., i : i + 1]
            if bmtk_compat:
                convolved_movie *= self.edge_reciprocals[i]
            flattened_movie = tf.reshape(
                convolved_movie[..., 0], (tf.shape(movie)[0], -1)
            )
            all_spatial_responses.append(
                _sample_spatial(
                    flattened_movie,
                    self.dominant_sample_indices[i],
                    self.dominant_sample_weights[i],
                )
            )
            all_non_dom_spatial_responses.append(
                _sample_spatial(
                    flattened_movie,
                    self.non_dominant_sample_indices[i],
                    self.non_dominant_sample_weights[i],
                )
            )

        all_spatial_responses = tf.concat(all_spatial_responses, axis=1)
        all_non_dom_spatial_responses = tf.concat(all_non_dom_spatial_responses, axis=1)
        all_spatial_responses = tf.gather(
            all_spatial_responses,
            self.sorted_neuron_ids_indices,
            axis=1,
        )
        all_non_dom_spatial_responses = tf.gather(
            all_non_dom_spatial_responses,
            self.composite_sort_indices,
            axis=1,
        )
        return all_spatial_responses, all_non_dom_spatial_responses

    @tf.function(jit_compile=True)
    def firing_rates_from_spatial(
        self, all_spatial_responses, all_non_dom_spatial_responses
    ):
        dom_filtered_output = LGN.temporal_filter(
            all_spatial_responses, self.dom_temporal_kernels
        )
        dom_firing_rates = LGN.transfer_function(
            dom_filtered_output * self.amplitude + self.spontaneous_firing_rates,
            dtype=self.dtype,
        )
        if self.n_composite == 0:
            return dom_firing_rates

        non_dom_filtered_output = LGN.temporal_filter(
            all_non_dom_spatial_responses,
            self.composite_non_dom_kernels,
        )
        composite_firing_rates = LGN.transfer_function(
            non_dom_filtered_output * self.composite_non_dom_amplitude
            + self.composite_spontaneous_rates,
            dtype=self.dtype,
        )
        non_dom_firing_rates = tf.transpose(
            tf.scatter_nd(
                self.composite_ids[:, None],
                tf.transpose(composite_firing_rates),
                (
                    tf.shape(all_spatial_responses)[1],
                    tf.shape(all_spatial_responses)[0],
                ),
            )
        )
        return dom_firing_rates + non_dom_firing_rates

    @staticmethod
    def create_one_unit_of_two_subunit_filter(prs, ttp_exp):
        filt = LGN.create_temporal_filter(prs)
        tcross_ind = get_tcross_from_temporal_kernel(
            filt.get_kernel(threshold=-1.0).kernel
        )
        filt_sum = filt.get_kernel(threshold=-1.0).kernel[:tcross_ind].sum()
        del_offset = ttp_exp - tcross_ind
        if del_offset >= 0:
            delays = prs["opt_delays"]
            delays[0] = delays[0] + del_offset
            delays[1] = delays[1] + del_offset
            prs["opt_delays"] = delays
            filt_new = LGN.create_temporal_filter(prs)
        else:
            logger.debug("del_offset < 0")
        return filt_new, filt_sum

    @staticmethod
    def create_temporal_filter(inp_dict):
        opt_wts = inp_dict["opt_wts"]
        opt_kpeaks = inp_dict["opt_kpeaks"]
        opt_delays = inp_dict["opt_delays"]
        return TemporalFilterCosineBump(opt_wts, opt_kpeaks, opt_delays)

    @staticmethod
    def transfer_function(arg__a, dtype=tf.float32):
        positive = tf.cast(arg__a >= 0, dtype)
        return positive * arg__a

    @staticmethod
    def select_spatial(x, y, convolved_movie):
        i1 = tf.cast(tf.stack([tf.floor(y), tf.floor(x)], axis=-1), tf.int32)
        i2 = tf.cast(tf.stack([tf.math.ceil(y), tf.floor(x)], axis=-1), tf.int32)
        i3 = tf.cast(tf.stack([tf.floor(y), tf.math.ceil(x)], axis=-1), tf.int32)
        i4 = tf.cast(tf.stack([tf.math.ceil(y), tf.math.ceil(x)], axis=-1), tf.int32)
        transposed = tf.transpose(convolved_movie, perm=[1, 2, 0])
        values = tf.stack(
            [
                tf.gather_nd(transposed, i1),
                tf.gather_nd(transposed, i2),
                tf.gather_nd(transposed, i3),
                tf.gather_nd(transposed, i4),
            ],
            axis=0,
        )
        y_factor = y - tf.floor(y)
        x_factor = x - tf.floor(x)
        weights = tf.stack(
            [
                (1 - x_factor) * (1 - y_factor),
                (1 - x_factor) * y_factor,
                x_factor * (1 - y_factor),
                x_factor * y_factor,
            ],
            axis=0,
        )
        return tf.einsum("int,in->tn", values, weights)

    @staticmethod
    def temporal_filter(all_spatial_responses, temporal_kernels):
        tr_spatial_responses = tf.pad(
            all_spatial_responses[None, :, None, :],
            ((0, 0), (temporal_kernels.shape[0] - 1, 0), (0, 0), (0, 0)),
        )
        return tf.nn.depthwise_conv2d(
            tr_spatial_responses,
            temporal_kernels[:, None, :, None],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )[0, :, 0]


def _stateless_seed_pair(seed, salt=0):
    if seed is None:
        return None
    max_int32 = 2**31 - 1
    seed_int = int(seed) % max_int32
    salt_int = int(salt) % max_int32
    return tf.constant([seed_int, (seed_int + salt_int) % max_int32], dtype=tf.int32)


def _fold_in_seed(seed_pair, value):
    with tf.device("/CPU:0"):
        return tf.random.experimental.stateless_fold_in(
            seed_pair, tf.cast(value, tf.int32)
        )


@tf.function
def movies_concat(movie, pre_delay, post_delay, dtype=tf.float32):
    # add an gray screen period before and after the movie
    z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    videos = tf.concat((z1, movie, z2), 0)
    return videos


# @tf.function # using jit_compile can cause error with input shapes
# def make_drifting_grating_stimulus(row_size=80, col_size=120, moving_flag=True, image_duration=100, cpd=0.05,
#                                    temporal_f=2, theta=0, phase=0, contrast=1.0, dtype=tf.float32):
#     '''
#     Create the grating movie with the desired parameters
#     :param t_min: start time in seconds
#     :param t_max: end time in seconds
#     :param cpd: cycles per degree
#     :param temporal_f: in Hz
#     :param theta: orientation angle
#     :return: Movie object of grating with desired parameters
#     '''
#     #  Franz's code will accept something larger than 101 x 101 because of the
#     #  kernel size.
#     # row_size = row_size*2 # somehow, Franz's code only accept larger size; thus, i did the mulitplication
#     # col_size = col_size*2
#     frame_rate = tf.constant(1000, dtype=dtype)  # Hz
#     # t_min = 0
#     # t_max = tf.cast(image_duration, tf.float32) / 1000
#     image_duration_f = tf.cast(image_duration, dtype=dtype)
#     pi = tf.constant(np.pi, dtype=dtype)

#     # assert contrast <= 1, "Contrast must be <= 1"
#     # assert contrast > 0, "Contrast must be > 0"
#     # tf.debugging.assert_less_equal(contrast, 1.0, message="Contrast must be <= 1")
#     # tf.debugging.assert_greater(contrast, 0.0, message="Contrast must be > 0")

#     # physical_spacing = 1. / (float(cpd) * 10)    #To make sure no aliasing occurs
#     # 1 degree per pixel; LGN x/y are in pixel coords [0, size-1], so avoid linspace endpoint overshoot.
#     # If you ever set physical_spacing != 1, you’ll need to rescale LGN x,y or change the stimulus grid size to keep alignment.
#     # Otherwise, the movie shape and LGN coordinates will no longer match.
#     physical_spacing = 1.0 # 1 degree, fixed for now. tf version lgn model need this to keep true cpd;
#     # row_range = tf.cast(tf.linspace(0.0, row_size, tf.cast(row_size / physical_spacing, tf.int32)), dtype=dtype)
#     # col_range = tf.cast(tf.linspace(0.0, col_size, tf.cast(col_size / physical_spacing, tf.int32)), dtype=dtype)
#     row_range = tf.cast(tf.range(0.0, tf.cast(row_size, dtype=tf.int32), delta=int(physical_spacing)), dtype=dtype)
#     col_range = tf.cast(tf.range(0.0, tf.cast(col_size, dtype=tf.int32), delta=int(physical_spacing)), dtype=dtype)
#     # number_frames_needed = int(round(frame_rate * t_max))
#     # number_frames_needed = tf.cast(tf.math.round(frame_rate * t_max), tf.int32)
#     # time_range = tf.cast(tf.linspace(0.0, t_max, number_frames_needed), dtype=dtype)
#     number_frames_needed = tf.cast(tf.math.round(image_duration_f), tf.int32)
#     time_range = tf.cast(tf.range(number_frames_needed), dtype=dtype) / frame_rate

#     tt, yy, xx = tf.meshgrid(time_range, row_range, col_range, indexing='ij')

#     # theta_rad = tf.constant(np.pi * (180 - theta) / 180.0, dtype=dtype) #Add negative here to match brain observatory angles!
#     # phase_rad = tf.constant(np.pi * (180 - phase) / 180.0, dtype=dtype)
#     theta_rad = pi * (180 - theta) / 180  # Convert to radians
#     phase_rad = pi * (180 - phase) / 180  # Convert to radians

#     xy = xx * tf.cos(theta_rad) + yy * tf.sin(theta_rad)
#     data = contrast * tf.sin(2 * pi * (cpd * xy + temporal_f * tt) + phase_rad)

#     if moving_flag: # decide whether the gratings drift or they are static
#         return data
#     else:
#         return tf.tile(data[0][tf.newaxis, ...], (image_duration, 1, 1))

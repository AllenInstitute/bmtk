import os
import six
import h5py
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings

# from ..core import SortOrder
# from ..spike_trains_api import SpikeTrainsReadOnlyAPI
# from ..core import col_node_ids, col_timestamps, col_population, pop_na, find_conversion, csv_headers
# from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version
# from bmtk.utils.io import bmtk_world_comm
#
#
# GRP_spikes_root = 'spikes'
# DATASET_timestamps = 'timestamps'
# DATASET_node_ids = 'node_ids'
#
#
# sorting_attrs = {
#     'time': SortOrder.by_time,
#     'by_time': SortOrder.by_time,
#     'id': SortOrder.by_id,
#     'by_id': SortOrder.by_id,
#     'none': SortOrder.none,
#     'unknown': SortOrder.unknown
# }
#

# def write_sonata(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, units='ms',
#                  population_renames=None, **kwargs):
#     rank = bmtk_world_comm.MPI_rank
#
#     path_dir = os.path.dirname(path)
#     if bmtk_world_comm == 0 and path_dir and not os.path.exists(path_dir):
#         os.makedirs(path_dir)
#
#     spiketrain_reader.flush()
#     bmtk_world_comm.barrier()
#
#     populations = spiketrain_reader.populations
#     spikes_root = None
#     if rank == 0:
#         h5 = h5py.File(path, mode=mode)
#         add_hdf5_magic(h5)
#         add_hdf5_version(h5)
#         spikes_root = h5.create_group('/spikes') if '/spikes' not in h5 else h5['/spikes']
#
#     for pop_name in populations: # metrics.keys():
#         if bmtk_world_comm.MPI_rank == 0 and pop_name in spikes_root:
#             # Problem if file already contains /spikes/<pop_name> # TODO: append new data to old spikes?!?
#             raise ValueError('sonata file {} already contains a spikes group {}, '.format(path, pop_name) +
#                              'skiping(use option mode="w" to overwrite)')
#
#         pop_df = spiketrain_reader.to_dataframe(populations=pop_name, with_population_col=False, sort_order=sort_order,
#                                                 on_rank='root')
#         if rank == 0:
#             spikes_pop_grp = spikes_root.create_group(pop_name)
#             if sort_order != SortOrder.unknown:
#                 spikes_pop_grp.attrs['sorting'] = sort_order.value
#
#             spikes_pop_grp.create_dataset('timestamps', data=pop_df['timestamps'])
#             spikes_pop_grp.create_dataset('node_ids', data=pop_df['node_ids'])
#     bmtk_world_comm.barrier()
#
#
# def write_sonata_itr(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, units='ms',
#                  population_renames=None, **kwargs):
#     path_dir = os.path.dirname(path)
#     if bmtk_world_comm.MPI_rank == 0 and path_dir and not os.path.exists(path_dir):
#         os.makedirs(path_dir)
#
#     spiketrain_reader.flush()
#     bmtk_world_comm.barrier()
#
#     conv_factor = find_conversion(spiketrain_reader.units, units)
#     if bmtk_world_comm.MPI_rank == 0:
#         h5 = h5py.File(path, mode=mode)
#         add_hdf5_magic(h5)
#         add_hdf5_version(h5)
#         spikes_root = h5.create_group('/spikes') if '/spikes' not in h5 else h5['/spikes']
#
#     population_renames = population_renames or {}
#     for pop_name in spiketrain_reader.populations:
#         n_spikes = spiketrain_reader.n_spikes(pop_name)
#         if n_spikes <= 0:
#             continue
#
#         # if pop_name in spikes_root:
#         #     raise ValueError('sonata file {} already contains a spikes group {}, '.format(path, pop_name) +
#         #                      'skiping(use option mode="w" to overwrite)')
#
#         if bmtk_world_comm.MPI_rank == 0:
#             spikes_grp = spikes_root.create_group('{}'.format(population_renames.get(pop_name, pop_name)))
#             if sort_order != SortOrder.unknown:
#                 spikes_grp.attrs['sorting'] = sort_order.value
#
#             timestamps_ds = spikes_grp.create_dataset('timestamps', shape=(n_spikes,), dtype=np.float64)
#             timestamps_ds.attrs['units'] = units
#             node_ids_ds = spikes_grp.create_dataset('node_ids', shape=(n_spikes,), dtype=np.uint64)
#
#         for i, spk in enumerate(spiketrain_reader.spikes(populations=pop_name, sort_order=sort_order)):
#             if bmtk_world_comm.MPI_rank == 0:
#                 timestamps_ds[i] = spk[0]*conv_factor
#                 node_ids_ds[i] = spk[2]
#
#     bmtk_world_comm.barrier()

#
# def load_sonata_file(path, version=None, **kwargs):
#     """Loads a Sonata file reader, making sure it matches the correct version.
#
#     :param path:
#     :param version:
#     :param kwargs:
#     :return:
#     """
#     try:
#         with h5py.File(path, 'r') as h5:
#             spikes_root = h5[GRP_spikes_root]
#             for name, h5_obj in spikes_root.items():
#                 if isinstance(h5_obj, h5py.Group):
#                     # In case there exists a population subgroup
#                     return SonataSTReader(path, **kwargs)
#     except Exception:
#         pass
#
#     try:
#         with h5py.File(path, 'r') as h5:
#             spikes_root = h5[GRP_spikes_root]
#             if 'gids' in spikes_root and 'timestamps' in spikes_root:
#                 return SonataOldReader(path, **kwargs)
#     except Exception:
#         pass
#
#     try:
#         with h5py.File(path, 'r') as h5:
#             if '/spikes' in h5:
#                 return EmptySonataReader(path, **kwargs)
#     except Exception:
#         pass
#
#     raise Exception('Could not open file {}, does not contain SONATA spike-trains'.format(path))
#
#
# def to_list(v):
#     if v is not None and np.isscalar(v):
#         return [v]
#     else:
#         return v
#
#
# class SonataSTReader(SpikeTrainsReadOnlyAPI):
#     def __init__(self, path, **kwargs):
#         self._path = path
#         self._h5_handle = h5py.File(self._path, 'r')
#         self._DATASET_node_ids = 'node_ids'
#         self._n_spikes = None
#         # TODO: Create a function for looking up population and can return errors if more than one
#         self._default_pop = None
#         # self._node_list = None
#
#         self._indexed = False
#         self._index_nids = {}
#
#         if GRP_spikes_root not in self._h5_handle:
#             raise Exception('Could not find /{} root'.format(GRP_spikes_root))
#         else:
#             self._spikes_root = self._h5_handle[GRP_spikes_root]
#
#         if 'population' in kwargs:
#             pop_filter = to_list(kwargs['population'])
#         elif 'populations' in kwargs:
#             pop_filter = to_list(kwargs['populations'])
#         else:
#             pop_filter = None
#
#         # get a map of 'pop_name' -> pop_group
#         self._population_map = {}
#         for name, h5_obj in self._h5_handle[GRP_spikes_root].items():
#             if isinstance(h5_obj, h5py.Group):
#                 if pop_filter is not None and name not in pop_filter:
#                     continue
#
#                 if 'node_ids' not in h5_obj or 'timestamps' not in h5_obj:
#                     warnings.warn('population {} in {} is missing spikes, skipping.'.format(name, path))
#                     continue
#
#                 self._population_map[name] = h5_obj
#
#         if not self._population_map:
#             # In old version of the sonata standard there was no 'population' subgroup. For backwards compatability
#             # use a default dictionary
#             # TODO: Remove so we only have to support latest version of SONATA
#             self._population_map = defaultdict(lambda: self._h5_handle[GRP_spikes_root])
#             self._population_map[pop_na] = self._h5_handle[GRP_spikes_root]
#             self._DATASET_node_ids = 'gids'
#
#         self._default_pop = kwargs.get('default_population', list(self._population_map.keys())[0])
#
#         self._population_sorting_map = {}
#         for pop_name, pop_grp in self._population_map.items():
#             if 'sorting' in pop_grp[self._DATASET_node_ids].attrs.keys():
#                 # Found a few existing sonata files put the 'sorting' attribute in the node_ids dataset, remove later
#                 attr_str = pop_grp[self._DATASET_node_ids].attrs['sorting']
#                 sort_order = sorting_attrs.get(attr_str, SortOrder.unknown)
#
#             elif 'sorting' in pop_grp.attrs.keys():
#                 attr_str = pop_grp.attrs['sorting']
#                 sort_order = sorting_attrs.get(attr_str, SortOrder.unknown)
#
#             else:
#                 sort_order = SortOrder.unknown
#
#             self._population_sorting_map[pop_name] = sort_order
#
#         # TODO: Add option to skip building indices
#         self._build_node_index()
#
#         # units are not instrinsic to a csv file, but allow users to pass it in if they know
#         # TODO: Should check the populations for the units
#         self._units = kwargs.get('units', 'ms')
#
#     def _build_node_index(self):
#         self._indexed = False
#         for pop_name, pop_grp in self._population_map.items():
#             sort_order = self._population_sorting_map[pop_name]
#             nodes_indices = {}
#             node_ids_ds = pop_grp[self._DATASET_node_ids]
#             if sort_order == SortOrder.by_id:
#                 indx_beg = 0
#                 last_id = node_ids_ds[0]
#                 for indx, cur_id in enumerate(node_ids_ds):
#                     if cur_id != last_id:
#                         # nodes_indices[last_id] = np.arange(indx_beg, indx)
#                         nodes_indices[last_id] = slice(indx_beg, indx)
#                         last_id = cur_id
#                         indx_beg = indx
#                 # nodes_indices[last_id] = np.arange(indx_beg, indx + 1)
#                 nodes_indices[last_id] = slice(indx_beg, indx + 1)  # capture the last node_id
#             else:
#                 nodes_indices = {int(node_id): [] for node_id in np.unique(node_ids_ds)}
#                 for indx, node_id in enumerate(node_ids_ds):
#                     nodes_indices[node_id].append(indx)
#
#             self._index_nids[pop_name] = nodes_indices
#
#         self._indexed = True
#
#     @property
#     def populations(self):
#         return list(self._population_map.keys())
#
#     @property
#     def units(self):
#         return self._units
#
#     @units.setter
#     def units(self, v):
#         self._units = v
#
#     def sort_order(self, population):
#         return self._population_sorting_map[population]
#
#     def node_ids(self, population=None):
#         population = population if population is not None else self._default_pop
#         pop_grp = self._population_map[population]
#         return np.unique(pop_grp[self._DATASET_node_ids][()])
#
#     def n_spikes(self, population=None):
#         population = population if population is not None else self._default_pop
#         return len(self._population_map[population][DATASET_timestamps])
#
#     def time_range(self, populations=None):
#         if populations is None:
#             populations = [self._default_pop]
#
#         if isinstance(populations, six.string_types) or np.isscalar(populations):
#             populations = [populations]
#
#         min_time = np.inf
#         max_time = -np.inf
#         for pop_name, pop_grp in self._population_map.items():
#             if pop_name in populations:
#                 # TODO: Check if sorted by time
#                 for ts in pop_grp[DATASET_timestamps]:
#                     if ts < min_time:
#                         min_time = ts
#
#                     if ts > max_time:
#                         max_time = ts
#
#         return min_time, max_time
#
#     def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
#         populations = populations if populations is not None else self.populations
#         if isinstance(populations, six.string_types) or np.isscalar(populations):
#             populations = [populations]
#
#         ret_df = None
#         for pop_name, pop_grp in self._population_map.items():
#             if pop_name in populations:
#                 pop_df = pd.DataFrame({
#                     col_timestamps: pop_grp[DATASET_timestamps],
#                     # col_population: pop_name,
#                     col_node_ids: pop_grp[self._DATASET_node_ids]
#                 })
#
#                 if with_population_col:
#                     pop_df['population'] = pop_name
#
#                 if sort_order == SortOrder.by_id:
#                     pop_df = pop_df.sort_values('node_ids')
#                 elif sort_order == SortOrder.by_time:
#                     pop_df = pop_df.sort_values('timestamps')
#
#                 ret_df = pop_df if ret_df is None else ret_df.append(pop_df)
#
#         if sort_order == SortOrder.by_time:
#             ret_df.sort_values(by=col_timestamps, inplace=True)
#         elif sort_order == SortOrder.by_id:
#             ret_df.sort_values(by=col_node_ids, inplace=True)
#
#         return ret_df
#
#     def get_times(self, node_id, population=None, time_window=None, **kwargs):
#         if population is None:
#             if not isinstance(self._default_pop, six.string_types) and len(self._default_pop) > 1:
#                 raise Exception('Error: Multiple populations, must select one.')
#
#             population = self._default_pop
#
#         elif population not in self._population_map:
#             return []
#
#         spikes_index = self._index_nids[population].get(node_id, None)
#         if spikes_index is None:
#             return []
#         spike_times = self._population_map[population][DATASET_timestamps][spikes_index]
#
#         if time_window is not None:
#             spike_times = spike_times[(time_window[0] <= spike_times) & (spike_times <= time_window[1])]
#
#         return spike_times
#
#     def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
#         populations = populations or self.populations
#         if np.isscalar(populations):
#             populations = [populations]
#
#         if sort_order == SortOrder.by_id:
#             for pop_name in populations:
#                 if pop_name not in self.populations:
#                     continue
#
#                 timestamps_ds = self._population_map[pop_name][DATASET_timestamps]
#                 index_map = self._index_nids[pop_name]
#                 node_ids = list(index_map.keys())
#                 node_ids.sort()
#                 for node_id in node_ids:
#                     st_indices = index_map[node_id]
#                     for st in timestamps_ds[st_indices]:  # st_indices:
#                         yield st, pop_name, node_id
#
#         elif sort_order == SortOrder.by_time:
#             # TODO: Reimplement using a heap
#             index_ranges = []
#             for pop_name in populations:
#                 if pop_name not in self.populations:
#                     continue
#
#                 pop_grp = self._population_map[pop_name]
#                 if self._population_sorting_map[pop_name] == SortOrder.by_time:
#                     ts_ds = pop_grp[DATASET_timestamps]
#                     index_ranges.append([pop_name, 0, len(ts_ds), np.arange(len(ts_ds)), pop_grp, ts_ds[0]])
#                 else:
#                     ts_ds = pop_grp[DATASET_timestamps]
#                     ts_indices = np.argsort(ts_ds[()])
#                     index_ranges.append([pop_name, 0, len(ts_ds), ts_indices, pop_grp, ts_ds[ts_indices[0]]])
#
#             while index_ranges:
#                 selected_r = index_ranges[0]
#                 for i, r in enumerate(index_ranges[1:]):
#                     if r[5] < selected_r[5]:
#                         selected_r = r
#
#                 ds_index = selected_r[1]
#                 timestamp = selected_r[5]  # pop_grp[DATASET_timestamps][ds_index]
#                 node_id = selected_r[4][self._DATASET_node_ids][ds_index]
#                 pop_name = selected_r[0]
#                 ds_index += 1
#                 if ds_index >= selected_r[2]:
#                     index_ranges.remove(selected_r)
#                 else:
#                     selected_r[1] = ds_index
#                     ts_index = selected_r[3][ds_index]
#                     next_ts = self._population_map[pop_name][DATASET_timestamps][selected_r[3][ds_index]]
#                     selected_r[5] = next_ts  # pop_grp[DATASET_timestamps][selected_r[3][ds_index]]
#
#                 yield timestamp, pop_name, node_id
#
#         else:
#             for pop_name in populations:
#                 if pop_name not in self.populations:
#                     continue
#
#                 pop_grp = self._population_map[pop_name]
#                 for i in range(len(pop_grp[DATASET_timestamps])):
#                     yield pop_grp[DATASET_timestamps][i], pop_name, pop_grp[self._DATASET_node_ids][i]
#
#     def __len__(self):
#         if self._n_spikes is None:
#             self._n_spikes = 0
#             for _, pop_grp in self._population_map.items():
#                 self._n_spikes += len(pop_grp[self._DATASET_node_ids])
#
#         return self._n_spikes
#
#
# class SonataOldReader(SonataSTReader):
#     """Older version of SONATA
#
#     """
#
#     def nodes(self, populations=None):
#         return super(SonataOldReader, self).nodes(populations=None)
#
#     def n_spikes(self, population=None):
#         return super(SonataOldReader, self).n_spikes(population=self._default_pop)
#
#     def time_range(self, populations=None):
#         return super(SonataOldReader, self).time_range(populations=None)
#
#     def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
#         return super(SonataOldReader, self).to_dataframe(node_ids=node_ids, populations=None,
#                                                          time_window=time_window, sort_order=sort_order, **kwargs)
#
#     def get_times(self, node_id, population=None, time_window=None, **kwargs):
#         return super(SonataOldReader, self).get_times(node_id=node_id, population=None,
#                                                       time_window=time_window, **kwargs)
#
#     def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
#         return super(SonataOldReader, self).spikes(node_ids=node_ids, populations=None,
#                                                    time_window=time_window, sort_order=sort_order, **kwargs)
#
#
# class EmptySonataReader(SpikeTrainsReadOnlyAPI):
#     """A Hack that is needed for when a simulation produces a file with no spikes, since there won't/can't be
#     <population_name> subgroup and/or gids/timestamps datasets.
#
#     """
#     def __init__(self, path, **kwargs):
#         pass
#
#     @property
#     def populations(self):
#         return []
#
#     def nodes(self, populations=None):
#         return []
#
#     def n_spikes(self, population=None):
#         return 0
#
#     def time_range(self, populations=None):
#         return None
#
#     def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
#         return pd.DataFrame(columns=csv_headers)
#
#     def get_times(self, node_id, population=None, time_window=None, **kwargs):
#         return []
#
#     def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
#         return []

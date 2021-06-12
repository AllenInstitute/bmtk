import numpy as np
import logging

logger = logging.getLogger(__name__)


class EdgesCollator(object):
    def __init__(self, edge_types_tables, sort_by=None):
        self._edge_types_tables = edge_types_tables

        self._model_groups_md = {}
        self._group_ids_lu = {}
        self._grp_id_itr = 0

        self.n_total_edges = sum(et.n_edges for et in edge_types_tables)

        self.assign_groups()

        self.source_ids = None
        self.target_ids = None
        self.edge_type_ids = None
        self.edge_group_ids = None
        self.edge_group_index = None
        self._prop_data = {}

    def merge_edges(self):
        self.source_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.target_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.edge_type_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_index = np.zeros(self.n_total_edges, dtype=np.uint32)

        self._prop_data = {
            g_id: {
                n: np.zeros(g_md['prop_size'], dtype=t) for n, t in zip(g_md['prop_names'], g_md['prop_type'])
            } for g_id, g_md in self._model_groups_md.items()
        }

        idx_beg = 0
        group_idx = {g_id: 0 for g_id in self._model_groups_md.keys()}
        for et in self._edge_types_tables:
            idx_end = idx_beg + et.n_edges

            src_trg_ids = et.prop_node_ids
            self.source_ids[idx_beg:idx_end] = src_trg_ids[:, 0]
            self.target_ids[idx_beg:idx_end] = src_trg_ids[:, 1]
            self.edge_type_ids[idx_beg: idx_end] = et.edge_type_id
            self.edge_group_ids[idx_beg: idx_end] = et.edge_group_id

            group_idx_beg = group_idx[et.edge_group_id]
            group_idx_end = group_idx_beg + et.n_edges
            self.edge_group_index[idx_beg: idx_end] = np.arange(group_idx_beg, group_idx_end, dtype=np.uint32)
            for pname, pdata in self._prop_data[et.edge_group_id].items():
                pdata[group_idx_beg:group_idx_end] = et.get_property_value(pname)

            idx_beg = idx_end
            group_idx[et.edge_group_id] += group_idx_end

            et.free_data()

    @property
    def group_ids(self):
        return list(self._prop_data.keys())

    def get_group_metadata(self, group_id):
        grp_md = self._model_groups_md[group_id]
        grp_dim = (grp_md['prop_size'], )
        return [{'name': n, 'type': t, 'dim': grp_dim} for n, t in zip(grp_md['prop_names'], grp_md['prop_type'])]

    def assign_groups(self):
        for et in self._edge_types_tables:
            # Assign each edge type a group_id based on the edge-type properties. When two edge-types tables use the
            # same model properties (in the hdf5) they should be put into the same group
            edge_types_hash = et.hash_key
            if edge_types_hash not in self._group_ids_lu:
                self._group_ids_lu[edge_types_hash] = self._grp_id_itr

                group_metadata = et.get_property_metatadata()
                self._model_groups_md[self._grp_id_itr] = {
                    'prop_names': [p['name'] for p in group_metadata],
                    'prop_type': [p['dtype'] for p in group_metadata],
                    'prop_size': 0
                }
                self._grp_id_itr += 1

            group_id = self._group_ids_lu[edge_types_hash]
            et.edge_group_id = group_id

            # number of rows in each model group
            self._model_groups_md[group_id]['prop_size'] += et.n_edges

    def itr_chunks(self):
        chunk_id = 0
        idx_beg = 0
        idx_end = self.n_total_edges

        yield chunk_id, idx_beg, idx_end

    def get_target_node_ids(self, chunk_id):
        return self.target_ids

    def get_source_node_ids(self, chunk_id):
        return self.source_ids

    def get_edge_type_ids(self, chunk_id):
        return self.edge_type_ids

    def get_edge_group_ids(self, chunk_id):
        return self.edge_group_ids

    def get_edge_group_indices(self, chunk_id):
        return self.edge_group_index

    def get_group_data(self, chunk_id):
        ret_val = []
        for group_id in self._prop_data.keys():
            for group_name in self._prop_data[group_id].keys():

                idx_end = len(self._prop_data[group_id][group_name])
                ret_val.append((group_id, group_name, 0, idx_end))

        return ret_val

    def get_group_property(self, group_name, group_id, chunk_id):
        return self._prop_data[group_id][group_name]

    def sort(self, sort_by, sort_group_properties=True):
        """In memory sort of the dataset

        :param sort_by:
        :param sort_group_properties:
        """
        # Find the edges hdf5 column to sort by
        if sort_by == 'target_node_id':
            sort_ds = self.target_ids
        elif sort_by == 'source_node_id':
            sort_ds = self.source_ids
        elif sort_by == 'edge_type_id':
            sort_ds = self.edge_type_ids
        elif sort_by == 'edge_group_id':
            sort_ds = self.edge_group_ids
        else:
            logger.warning('Unable to sort dataset, unrecognized column {}.'.format(sort_by))
            return

        # check if dataset is already sorted
        if np.all(np.diff(sort_ds) >= 0):
            return

        # Find order of arguments of sorted arrays, and sort main columns
        sort_idx = np.argsort(sort_ds)
        self.source_ids = self.source_ids[sort_idx]
        self.target_ids = self.target_ids[sort_idx]
        self.edge_type_ids = self.edge_type_ids[sort_idx]
        self.edge_group_ids = self.edge_group_ids[sort_idx]
        self.edge_group_index = self.edge_group_index[sort_idx]

        if sort_group_properties:
            # For sorting group properties, so the "edge_group_index" column is sorted (wrt each edge_group_id). Note
            # that it is not strictly necessary to sort the group properties, as sonata edge_group_index keeps the
            # reference, but doing the sorting may improve memory/efficency during setting up a simulation

            for grp_id, grp_props in self._prop_data.items():
                # Filter out edge_group_index array for each group_id, get the new order and apply to each property.
                grp_id_filter = np.argwhere(self.edge_group_ids == grp_id).flatten()
                updated_order = self.edge_group_index[grp_id_filter]
                for prop_name, prop_data in grp_props.items():
                    grp_props[prop_name] = prop_data[updated_order]

                # reorder the edge_group_index (for values with corresponding group_id)
                self.edge_group_index[grp_id_filter] = np.arange(0, len(grp_id_filter), dtype=np.uint32)

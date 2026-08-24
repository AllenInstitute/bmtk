import numpy as np
import pandas as pd


class TFIDMap:
    """Main class used to map SONATA node_ids to tensorflow neuron ids (tf-ids)
    
    This is a singleton class, meaning that for the lifespan of the Python interpreter run the mapping 
    between node-ids and tf-ids will not change. 
    TODO: Consider using Multion pattern that can be tied to each RNN instance.
    
    Things this class must take care of.
    1. tf-ids must start with 0 and be contigious. SONATA node_ids must be positive non-overlapping intergers,
       but may have gaps and can be out-of-order.
    2. Recurrent nodes (eg. glif cells) must not have overlapping tf-ids. However in SONATA it is possible for
       a recurrent network to be made up of multiple populations with overlapping node-ids.
    3. Each input network (eg. virtual cell) must have tf-ids [0, n_input_cells]. As above this is not 
       required for SONATA.

    """

    _tf_id_map_instance = None
    _initialized = False
    def __new__(cls):
        if cls._tf_id_map_instance is None:
            cls._tf_id_map_instance = super().__new__(cls)
        return cls._tf_id_map_instance
    
    def __init__(self):
        if not self._initialized:
            self._bmtk_populations = {}
            self._recurrent_tf_indices = [0]
            self._recurrent_populations = []
            self._initialized = True

    def add_bmtk_ids(self, population_name, node_ids, network_type):
        if population_name in self._bmtk_populations:
            raise ValueError(f'Attempting to create multiple bmtk -> tf-id maps for node population "{population_name}".')
        
        if len(node_ids) == 0:
            raise ValueError(f'Attempting to create a bmtk -> tf-id map with no-nodes')

        if network_type == 'recurrent':
            # If there are multiple recurrent networks we must make sure the tf ids don't overlap. ex. If network A has 50 nodes
            # and network B has 100 nodes, the bmtk2tf_map[A] = [0, 1, ..., 49], and bmtk2tf_map[B] = [50, 51, ..., 149]
            tf_idx_beg = self._recurrent_tf_indices[-1]
            self._recurrent_tf_indices.append(tf_idx_beg + len(node_ids))
            self._recurrent_populations.append(population_name)
            tf_idx_end = self._recurrent_tf_indices[-1]
        elif network_type == 'input':
            # For a input network the bmtk2tf_map = [0, 1, 2, ..., n_nodes]
            tf_idx_beg, tf_idx_end = 0, len(node_ids)
        else:
            raise ValueError(f'Unknown network_type {network_type}')

        # BMTK node_ids must be non-negative integers, but don't have to be sequential, ex. if map bmtk -> tf is 
        # [0, 3, 6, ...] -> [0, 1, 2, ...], then map is an array of form [0, NA, NA, 1, NA, NA, 2, ...] 
        max_id = np.max(node_ids)
        bmtk2tf_id_map = np.full(int(max_id)+1, -1, dtype=np.int64)
        bmtk2tf_id_map[node_ids] = np.arange(tf_idx_beg, tf_idx_end, dtype=np.int64)

        self._bmtk_populations[population_name] = {
            'tf_ids': (tf_idx_beg, tf_idx_end),
            # 'bmtk_ids': node_ids, # TODO: Should a copy be made?
            'bmtk2tf_id_map': bmtk2tf_id_map
        }

    def bmtk2tf_id_map(self, population_name):
        if population_name not in self._bmtk_populations:
            raise ValueError(f'No population "{population_name}" found.')
        
        return self._bmtk_populations[population_name]['bmtk2tf_id_map']

    def recurrent_bmtk_ids(self, as_type='dict'):
        bmtk_ids_dict = {}
        for pop in self._recurrent_populations:
            bmtk_ids_dict[pop] = self._bmtk_populations[pop]['bmtk2tf_id_map']

        return bmtk_ids_dict
    
    def tf2bmtk_id_map(self, populations=None):
        if populations is None or populations == 'recurrent':
            pop_list = self._recurrent_populations
        elif isinstance(populations, str):
            pop_list = self._bmtk_populations[populations]
        else:
            pop_list = [p for p in self._bmtk_populations if p in populations]

        assert(len(pop_list) > 0)
        ret_df = None
        for pop in pop_list:
            pop_props = self._bmtk_populations[pop]
            tmp_df = pd.DataFrame({
                'tf_ids': range(pop_props['tf_ids'][0], pop_props['tf_ids'][1]),
                'node_id': pop_props['bmtk2tf_id_map'],
                'population': pop
            })
            if ret_df is None:
                ret_df = tmp_df
            else:
                ret_df = pd.concat([ret_df, tmp_df])

        return ret_df.set_index('tf_ids')

    def reset(self):
        self._bmtk_populations = {}
        self._recurrent_tf_indices = [0]
        self._recurrent_populations = []
        self._initialized = True

from pathlib import Path
import h5py
import pandas as pd
import numpy as np

from .sim_module import SimulatorMod
from bmtk.simulator.core.io_tools import io


class RatesRecorderMod(SimulatorMod):
    def __init__(self, name, output_type, module, file_name, **kwargs):
        self._name = name
        self._output_type = output_type
        self._module = module
        self._params = kwargs
        self.file_name = file_name
        self._output_dir = kwargs.get('output_dir', '.')
        self._node_set = kwargs.get('cells', None)
                

        self.file_path = Path(self.file_name)
        if not self.file_path.is_absolute():
            self.file_path = (Path(self._output_dir) / self.file_path).absolute()

        if self._output_type not in ['csv', 'h5']:
            raise ValueError(f'{self.__class__.__name__}: Invalid output_type "{self._output_type}". [Valid options: csv, h5]')

        self._include_columns = kwargs.get('include_columns', [])
        self._include_columns = [self._include_columns] if not isinstance(self._include_columns, (list, tuple)) else self._include_columns
        self._include_columns = [k for k in self._include_columns if k not in ['population', 'node_id', 'firing_rates', 'timestamps']]

    def finalize(self, sim):
        if self._output_type == 'csv':
            self._to_csv(sim)
        elif self._output_type == 'h5':
            self._to_hdf5(sim)

    def _get_nodes(self, sim):
        if self._node_set is not None:
            node_set = sim.network.get_node_set(self._node_set)
            return [sim.network.get_node(n.population_name, n.node_id) for n in node_set.fetch_nodes()]
        else:
            return list(sim.network._ssn_recurrent_nodes)

    def _get_timestamps(self, sim):
        return np.linspace(0.0, sim.tstop, num=sim.nsteps, endpoint=False)

    def _to_csv(self, sim):
        
        output_df = None
        times = self._get_timestamps(sim)
        ssn_nodes = self._get_nodes(sim)

        for node in ssn_nodes:
            tmp_df = pd.DataFrame({
                'population': node.population,
                'node_id': node.node_id,
                'timestamps': times,
                'firing_rates': sim.results[:, node.gid]
            })

            for c in self._include_columns:
                tmp_df[c] = node.get(c, None)

            
            output_df = tmp_df if output_df is None else pd.concat([output_df, tmp_df], ignore_index=True)

        if output_df is not None:
            io.log_info(f'{self.__class__.__name__}: Saving rates to {self.file_path}.')
            output_df.to_csv(self.file_path, sep=' ', index=False)
        

    def _to_hdf5(self, sim):
        mode = self._params.get('mode', 'a')
        times = self._get_timestamps(sim)
        nsteps = len(times)
        ssn_nodes = self._get_nodes(sim)
        mappings = {}
        for n in ssn_nodes:
            subpop = mappings.get(n.population, {k: [] for k in ['node_id', 'gid'] + self._include_columns})
            subpop['node_id'].append(n.node_id)
            subpop['gid'].append(n.gid)
            for c in self._include_columns:
                subpop[c].append(n.get(c, None))
            
            mappings[n.population] = subpop

        with h5py.File(self.file_path, mode) as h5:
            io.log_info(f'{self.__class__.__name__}: Saving rates to {self.file_path}.')
            ratesgrp = h5['rates'] if 'rates' in h5 else h5.create_group('rates')
            for pop_name, pop_data in mappings.items():
                subgrp = ratesgrp.create_group(pop_name)
                subgrp.create_dataset('mapping/time', data=times)
                subgrp.create_dataset('mapping/node_ids', data=pop_data['node_id'])
                for c in self._include_columns:
                    subgrp.create_dataset(f'mapping/{c}', data=pop_data[c])

                data = subgrp.create_dataset('data', shape=(nsteps, len(pop_data['gid'])), dtype=float)
                for col, gid in enumerate(pop_data['gid']):
                    data[:, col] = sim.results[:, gid]


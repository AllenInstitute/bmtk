import h5py
import numpy as np


class CellVarsFile(object):
    def __init__(self, filename, mode='r', **params):
        self._h5_handle = h5py.File(filename, 'r')
        self._var_data = {}
        self._mapping = None

        # Look for variabl and mapping groups
        for var_name in self._h5_handle.keys():
            hf_grp = self._h5_handle[var_name]
            if not isinstance(hf_grp, h5py.Group):
                continue

            if var_name == 'mapping':
                self._mapping = hf_grp
            else:
                if 'data' not in hf_grp:
                    print('Warning: could not find "data" dataset in {}. Skipping!'.format(var_name))
                else:
                    self._var_data[var_name] = hf_grp['data']

        # create map between gids and tables
        self._gid2data_table = {}
        if self._mapping == None:
            raise Exception('could not find /mapping group')
        else:
            gids_ds = self._mapping['gids']
            index_pointer_ds = self._mapping['index_pointer']
            for indx, gid in enumerate(gids_ds):
                self._gid2data_table[gid] = (index_pointer_ds[indx], index_pointer_ds[indx+1])  # slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

            time_ds = self._mapping['time']
            self._t_start = time_ds[0]
            self._t_stop = time_ds[1]
            self._dt = time_ds[2]
            self._n_steps = int((self._t_stop - self._t_start) / self._dt)

    @property
    def variables(self):
        return list(self._var_data.keys())

    @property
    def gids(self):
        return list(self._gid2data_table.keys())

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def dt(self):
        return self._dt

    @property
    def time_trace(self):
        return np.linspace(self.t_start, self.t_stop, num=self._n_steps, endpoint=True)

    def n_compartments(self, gid):
        bounds = self._gid2data_table[gid]
        return bounds[1] - bounds[0]

    def compartment_ids(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_id'][bounds[0]:bounds[1]]

    def compartment_positions(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_pos'][bounds[0]:bounds[1]]

    def data(self, var_name, gid, time_window=None, compartments='origin'):
        if var_name not in self.variables:
            raise Exception('Unknown variable {}'.format(var_name))

        if time_window is None:
            time_slice = slice(0, self._n_steps)
        else:
            window_beg = max(int(time_window[0]/self.dt), 0)
            window_end = min(int(time_window[1]/self.dt), self._n_steps)
            time_slice = slice(window_beg, window_end)

        multi_compartments = True
        if compartments == 'origin' or self.n_compartments(gid) == 1:
            # Return the first (and possibly only) compartment for said gid
            gid_slice = self._gid2data_table[gid][0]
            multi_compartments = False
        elif compartments == 'all':
            # Return all compartments
            gid_slice = slice(self._gid2data_table[gid][0], self._gid2data_table[gid][1])
        else:
            # return all compartments with corresponding element id
            compartment_list = list(compartments) if isinstance(compartments, (long, int)) else compartments
            begin = self._gid2data_table[gid][0]
            end = self._gid2data_table[gid][1]
            gid_slice = [i for i in range(begin, end) if self._mapping[i] in compartment_list]

        data = np.array(self._var_data[var_name][time_slice, gid_slice])
        return data.T if multi_compartments else data

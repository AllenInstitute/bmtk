import h5py
import numpy as np

from .core import CompartmentReaderABC


class _CompartmentPopulationReaderVer01(CompartmentReaderABC):
    sonata_columns = ['element_ids', 'element_pos', 'index_pointer', 'node_ids', 'time']

    def __init__(self, pop_grp, pop_name):
        self._data_grp = pop_grp['data']
        self._mapping = pop_grp['mapping']
        self._population = pop_name

        self._gid2data_table = {}
        if self._mapping is None:
            raise Exception('could not find /mapping group')

        gids_ds = self._mapping['node_ids']
        index_pointer_ds = self._mapping['index_pointer']
        for indx, gid in enumerate(gids_ds):
            self._gid2data_table[gid] = slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

        time_ds = self._mapping['time']
        self._t_start = np.float(time_ds[0])
        self._t_stop = np.float(time_ds[1])
        self._dt = np.float(time_ds[2])
        self._n_steps = int((self._t_stop - self._t_start) / self._dt)

        self._custom_cols = {col: grp for col, grp in self._mapping.items() if
                             col not in self._mapping and isinstance(grp, h5py.Dataset)}

    def _get_index(self, node_id):
        return self._gid2data_table[node_id]

    @property
    def populations(self):
        return [self._population]

    @property
    def data_ds(self):
        return self._data_grp

    def get_population(self, population, default=None):
        raise NotImplementedError()

    def units(self, population=None):
        return self.data_ds.attrs.get('units', None)

    def variable(self, population=None):
        return self.data_ds.attrs.get('variable', None)

    def t_start(self, population=None):
        return self._t_start

    def t_stop(self, population=None):
        return self._t_stop

    def dt(self, population=None):
        return self._dt

    def n_steps(self, population=None):
        return self._n_steps

    def time_trace(self, population=None):
        return np.linspace(self.t_start(), self.t_stop(), num=self._n_steps, endpoint=True)

    def node_ids(self, population=None):
        return self._mapping['node_ids'][()]

    def element_pos(self, node_id=None, population=None):
        if node_id is None:
            return self._mapping['element_pos'][()]
        else:
            return self._mapping['element_pos'][self._get_index(node_id)]#[indx_beg:indx_end]

    def element_ids(self, node_id=None, population=None):
        if node_id is None:
            return self._mapping['element_ids'][()]
        else:
            indx_beg, indx_end = self._get_index(node_id)
            return self._mapping['element_ids'][self._get_index(node_id)]#[indx_beg:indx_end]

    def n_elements(self, node_id=None, population=None):
        return len(self.element_pos(node_id))

    def data(self, node_id=None, population=None, time_window=None, sections='all', **opts):
        # filtered_data = self._data_grp
        multi_compartments = True
        if node_id is not None:
            node_range = self._get_index(node_id)
            if sections == 'origin' or self.n_elements(node_id) == 1:
                # Return the first (and possibly only) compartment for said gid
                gid_slice = node_range
                multi_compartments = False
            elif sections == 'all':
                # Return all compartments
                gid_slice = node_range #slice(node_beg, node_end)
            else:
                # return all compartments with corresponding element id
                compartment_list = list(sections) if np.isscalar(sections) else sections
                gid_slice = [i for i in self._get_index(node_id) if self._mapping['element_ids'] in compartment_list]
        else:
            gid_slice = slice(0, self._data_grp.shape[1])

        if time_window is None:
            time_slice = slice(0, self._n_steps)
        else:
            if len(time_window) != 2:
                raise Exception('Invalid time_window, expecting tuple [being, end].')

            window_beg = max(int((time_window[0] - self.t_start) / self.dt), 0)
            window_end = min(int((time_window[1] - self.t_start) / self.dt), self._n_steps / self.dt)
            time_slice = slice(window_beg, window_end)

        filtered_data = np.array(self._data_grp[time_slice, gid_slice])
        return filtered_data.T if multi_compartments else filtered_data


    def custom_columns(self, population=None):
        pass

    def get_column(self, column_name, population=None):
        pass

    def get_node_description(self, node_id, population=None):
        pass

    def get_report_description(self, population=None):
        pass

    def __getitem__(self, population):
        return self


class CompartmentReaderVer01(CompartmentReaderABC):
    def __init__(self, filename, mode='r', **params):
        self._h5_handle = h5py.File(filename, mode)
        self._h5_root = self._h5_handle[params['h5_root']] if 'h5_root' in params else self._h5_handle['/']
        self._popgrps = {}

        self._mapping = None

        if 'report' in self._h5_handle.keys():
            report_grp = self._h5_root['report']
            for pop_name, pop_grp in report_grp.items():
                self._popgrps[pop_name] = _CompartmentPopulationReaderVer01(pop_grp=pop_grp, pop_name=pop_name)

        if 'default_population' in params:
            # If user has specified a default population
            self._default_population = params['default_population']
            if self._default_population not in self._popgrps.keys():
                raise Exception('Unknown population {} found in report.'.format(self._default_population))
        elif len(self._popgrps.keys()) == 1:
            # If there is only one population in the report default to that
            self._default_population = list(self._popgrps.keys())[0]
        else:
            self._default_population = None

    @property
    def default_population(self):
        if self._default_population is None:
            raise Exception('Please specify a node population.')
        return self._default_population

    @property
    def populations(self):
        return list(self._popgrps.keys())

    def get_population(self, population, default=None):
        if population not in self.populations:
            return default
        return self[population]

    def units(self, population=None):
        population = population or self.default_population
        return self[population].units()

    def variable(self, population=None):
        population = population or self.default_population
        return self[population].variable()

    def t_start(self, population=None):
        population = population or self.default_population
        return self[population].t_start()

    def t_stop(self, population=None):
        population = population or self.default_population
        return self[population].t_stop()

    def dt(self, population=None):
        population = population or self.default_population
        return self[population].dt()

    def time_trace(self, population=None):
        population = population or self.default_population
        return self[population].time()

    def node_ids(self, population=None):
        population = population or self.default_population
        return self[population].node_ids()

    def element_pos(self, node_id=None, population=None):
        population = population or self.default_population
        return self[population].element_pos(node_id)

    def element_ids(self, node_id=None, population=None):
        population = population or self.default_population
        return self[population].element_ids(node_id)

    def n_elements(self, node_id=None, population=None):
        population = population or self.default_population
        return self[population].n_elements(node_id)

    def data(self, node_ids=None, population=None, time_window=None, sections='all', **opt_attrs):
        population = population or self.default_population
        return self[population].data(node_ids=node_ids, time_window=time_window, sections=sections, **opt_attrs)

    def custom_columns(self, population=None):
        population = population or self.default_population
        return self[population].custom_columns(population)

    def get_column(self, column_name, population=None):
        population = population or self.default_population
        return self[population].get_column(column_name)

    def get_node_description(self, node_id, population=None):
        population = population or self.default_population
        return self[population].get_node_description(node_id)

    def get_report_description(self, population=None):
        population = population or self.default_population
        return self[population].get_report_description()

    def __getitem__(self, population):
        return self._popgrps[population]




'''
class PopulationReader(CompartmentReaderABC):
    mapping_ds = ['element_ids', 'element_pos', 'index_pointer', 'node_ids', 'time']

    def __init__(self, grp, population):
        self._data_grp = grp['data']
        self._mapping = grp['mapping']

        self._gid2data_table = {}
        if self._mapping is None:
            raise Exception('could not find /mapping group')
        else:
            gids_ds = self._mapping['node_ids']
            index_pointer_ds = self._mapping['index_pointer']
            for indx, gid in enumerate(gids_ds):
                self._gid2data_table[gid] = (index_pointer_ds[indx], index_pointer_ds[
                    indx + 1])  # slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

            time_ds = self._mapping['time']
            self._t_start = time_ds[0]
            self._t_stop = time_ds[1]
            self._dt = time_ds[2]
            self._n_steps = int((self._t_stop - self._t_start) / self._dt)

        self._custom_cols = {col: grp for col, grp in self._mapping.items() if
                             col not in self.mapping_ds and isinstance(grp, h5py.Dataset)}

    @property
    def element_pos(self):
        return self._mapping['element_pos'][()]

    @property
    def element_ids(self):
        return self._mapping['element_ids'][()]

    @property
    def node_ids(self):
        return self._mapping['node_ids'][()]

    @property
    def time(self):
        return self._mapping['time'][()]

    @property
    def index_pointer(self):
        return self._mapping['index_pointer'][()]

    @property
    def custom_columns(self):
        return self._custom_cols

    @property
    def n_steps(self):
        return len(self._data_grp)

    @property
    def data(self):
        return self._data_grp

    @property
    def variable(self):
        return self.data.attrs.get('variable', None)

    @property
    def units(self):
        return self.data.attrs.get('units', None)
'''


'''
class CompartmentReader(object):
    VAR_UNKNOWN = 'Unknown'
    UNITS_UNKNOWN = 'NA'

    def __init__(self, filename, mode='r', **params):
        self._h5_handle = h5py.File(filename, 'r')
        self._h5_root = self._h5_handle[params['h5_root']] if 'h5_root' in params else self._h5_handle['/']
        #self._var_data = {}
        #self._var_units = {}
        self._popgrps = {}

        self._mapping = None

        if 'report' in self._h5_handle.keys():
            report_grp = self._h5_handle['report']
            for pop_name, pop_grp in report_grp.items():
                self._popgrps[pop_name] = PopulationReader(pop_grp, pop_name)


        """
        # Look for variabl and mapping groups
        for var_name in self._h5_root.keys():
            hf_grp = self._h5_root[var_name]

            if var_name == 'data':
                # According to the sonata format the /data table should be located at the root
                var_name = self._h5_root['data'].attrs.get('variable_name', CompartmentReader.VAR_UNKNOWN)
                self._var_data[var_name] = self._h5_root['data']
                self._var_units[var_name] = self._find_units(self._h5_root['data'])

            if not isinstance(hf_grp, h5py.Group):
                continue

            if var_name == 'mapping':
                # Check for /mapping group
                self._mapping = hf_grp
            else:
                # In the bmtk we can support multiple variables in the same file (not sonata compliant but should be)
                # where each variable table is separated into its own group /<var_name>/data
                if 'data' not in hf_grp:
                    print('Warning: could not find "data" dataset in {}. Skipping!'.format(var_name))
                else:
                    self._var_data[var_name] = hf_grp['data']
                    self._var_units[var_name] = self._find_units(hf_grp['data'])

        """

        """
        # create map between gids and tables
        self._gid2data_table = {}
        if self._mapping is None:
            raise Exception('could not find /mapping group')
        else:
            gids_ds = self._mapping['gids']
            index_pointer_ds = self._mapping['index_pointer']
            for indx, gid in enumerate(gids_ds):
                self._gid2data_table[gid] = (index_pointer_ds[indx], index_pointer_ds[
                    indx + 1])  # slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

            time_ds = self._mapping['time']
            self._t_start = time_ds[0]
            self._t_stop = time_ds[1]
            self._dt = time_ds[2]
            self._n_steps = int((self._t_stop - self._t_start) / self._dt)
        """

    @property
    def populations(self):
        return self._popgrps.keys()

    def __getitem__(self, population):
        return self._popgrps[population]

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

    @property
    def h5(self):
        return self._h5_root

    def _find_units(self, data_set):
        return data_set.attrs.get('units', CompartmentReader.UNITS_UNKNOWN)

    def units(self, var_name=VAR_UNKNOWN):
        return self._var_units[var_name]

    def n_compartments(self, gid):
        bounds = self._gid2data_table[gid]
        return bounds[1] - bounds[0]

    def compartment_ids(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_id'][bounds[0]:bounds[1]]

    def compartment_positions(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_pos'][bounds[0]:bounds[1]]

    def data(self, gid, var_name=VAR_UNKNOWN, time_window=None, compartments='origin'):
        if var_name not in self.variables:
            raise Exception('Unknown variable {}'.format(var_name))

        if time_window is None:
            time_slice = slice(0, self._n_steps)
        else:
            if len(time_window) != 2:
                raise Exception('Invalid time_window, expecting tuple [being, end].')

            window_beg = max(int((time_window[0] - self.t_start) / self.dt), 0)
            window_end = min(int((time_window[1] - self.t_start) / self.dt), self._n_steps / self.dt)
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

    def close(self):
        self._h5_handle.close()
'''
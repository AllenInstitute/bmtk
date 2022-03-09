import os
import glob
import pandas as pd
from bmtk.utils.reports import CompartmentReport
from bmtk.simulator.pointnet.io_tools import io
from bmtk.simulator.pointnet.nest_utils import nest_version

import nest


try:
    MPI_RANK = nest.Rank()
    N_HOSTS = nest.NumProcesses()

except Exception as e:
    MPI_RANK = 0
    N_HOSTS = 1


def create_multimeter_nest2(tstart, tstop, variable_name, label):
    return nest.Create(
        'multimeter',
        params={
            'start': tstart,
            'stop': tstop,
            'to_file': True,
            'to_memory': False,
            'withtime': True,
            'record_from': variable_name,
            'label': label
        }
    )


def create_multimeter_nest3(tstart, tstop, variable_name, label):
    return nest.Create(
        'multimeter',
        params={
            'start': tstart,
            'stop': tstop,
            'record_to': 'ascii',
            'record_from': variable_name,
            'label': label
        }
    )


def read_dat_nest2(dat_file, variable_name):
    return pd.read_csv(dat_file, index_col=False, names=['nest_id', 'time']+variable_name, sep='\t')


def read_dat_nest3(dat_file, variable_name):
    report_df = pd.read_csv(dat_file, index_col=False, sep='\t', comment='#')
    report_df = report_df.rename(columns={'sender': 'nest_id', 'time_ms': 'time'})
    return report_df



if nest_version[0] >= 3:
    create_multimeter = create_multimeter_nest3
    read_dat = read_dat_nest3
else:
    create_multimeter = create_multimeter_nest2
    read_dat = read_dat_nest2


class MultimeterMod(object):
    def __init__(self, tmp_dir, file_name, variable_name, cells, tstart=None, tstop=None, interval=None, to_h5=True,
                 delete_dat=True, **opt_params):
        """For recording neuron properties using a NEST multimeter object

        :param tmp_dir: ouput directory
        :param file_name: Name of (SONATA hdf5) file that will be saved to
        :param variable_name: A list of the variable(s) being recorded. Must be valid according to the cells
        :param cells: A node-set or list of gids to record from
        :param tstart: Start time of the recording (if None will default to sim.tstart)
        :param tstop: Stop time of recording (if None will default to sim.tstop)
        :param interval: Recording time step (if None will default to sim.dt)
        :param to_h5: True to save to sonata .h5 format (default: True)
        :param delete_dat: True to delete the .dat files created by NEST (default True)
        :param opt_params:
        """
        self._output_dir = tmp_dir
        self._file_name = file_name if os.path.isabs(file_name) else os.path.join(self._output_dir, file_name)
        self._variable_name = variable_name
        self._node_set = cells
        self._tstart = tstart
        self._tstop = tstop
        self._interval = interval
        self._to_h5 = to_h5
        self._delete_dat = delete_dat

        self._gids = None  # global ids will be the NEST ids assigned to each cell
        self._multimeter = None
        self._population = None

        self._min_delay = 1.0  # Required for calculating steps recorded

        self.__output_label = os.path.join(self._output_dir, '__bmtk_nest_{}'.format(os.path.basename(self._file_name)))
        self._var_recorder = None  # CellVarRecorder(self._file_name, self._output_dir, self._variable_name, buffer_data=False)

    def initialize(self, sim):
        node_set = sim.net.get_node_set(self._node_set)

        self._gids = list(set(node_set.gids()))
        self._gids.sort()
        self._population = node_set.population_names()[0]
        self._tstart = self._tstart or sim.tstart
        self._tstop = self._tstop or sim.tstop
        self._interval = self._interval or sim.dt
        self._multimeter = create_multimeter(self._tstart, self._tstop, self._variable_name, self.__output_label)

        nest.SetStatus(self._multimeter, 'interval', self._interval)
        nest.Connect(self._multimeter, self._gids)

    def finalize(self, sim):
        io.barrier()  # Makes sure all nodes finish, but not sure if actually required by nest

        # min_delay needs to be fetched after simulation otherwise the value will be off. There also seems to be some
        # MPI barrier inside GetKernelStatus
        self._min_delay = nest.GetKernelStatus('min_delay')
        if self._to_h5 and MPI_RANK == 0:
            # Initialize hdf5 file including preallocated data block of recorded variables
            #   Unfortantely with NEST the final time-step recorded can't be calculated in advanced, and even with the
            # same min/max_delay can be different. We need to read the output-file to get n_steps
            def get_var_recorder(node_recording_df):
                if self._var_recorder is None:
                    self._var_recorder = CompartmentReport(self._file_name, mode='w', variable=self._variable_name[0],
                                                           default_population=self._population,
                                                           tstart=node_recording_df['time'].min(),
                                                           tstop=node_recording_df['time'].max(),
                                                           dt=self._interval,
                                                           n_steps=len(node_recording_df),
                                                           mpi_size=1)
                    if self._to_h5 and MPI_RANK == 0:
                        for gid in self._gids:
                            pop_id = gid_map.get_pool_id(gid)
                            self._var_recorder.add_cell(pop_id.node_id, element_ids=[0], element_pos=[0.0],
                                                        population=pop_id.population)

                    self._var_recorder.initialize()

                return self._var_recorder

            gid_map = sim.net.gid_map
            for nest_file in glob.glob('{}*'.format(self.__output_label)):
                # report_df = pd.read_csv(nest_file, index_col=False, names=['nest_id', 'time']+self._variable_name,
                #                         sep='\t', comment='#')
                report_df = read_dat(nest_file, self._variable_name)
                # print(report_df)
                # exit()

                for grp_id, grp_df in report_df.groupby(by='nest_id'):
                    pop_id = gid_map.get_pool_id(grp_id)
                    vr = get_var_recorder(grp_df)
                    for var_name in self._variable_name:
                        vr.record_cell_block(node_id=pop_id.node_id, vals=grp_df[var_name], beg_step=0,
                                             end_step=vr[pop_id.population].n_steps(), population=pop_id.population)

                if self._delete_dat:
                    # remove csv file created by nest
                    os.remove(nest_file)

            self._var_recorder.close()

        io.barrier()

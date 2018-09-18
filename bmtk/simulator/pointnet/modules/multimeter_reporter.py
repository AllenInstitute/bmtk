import os
import glob
import pandas as pd
from bmtk.utils.io.cell_vars import CellVarRecorder
from bmtk.simulator.pointnet.io_tools import io

import nest


try:
    MPI_RANK = nest.Rank()
    N_HOSTS = nest.NumProcesses()

except Exception as e:
    MPI_RANK = 0
    N_HOSTS = 1


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

        self._gids = None
        self._nest_ids = None
        self._multimeter = None

        self._min_delay = 1.0  # Required for calculating steps recorded

        self.__output_label = os.path.join(self._output_dir, '__bmtk_nest_{}'.format(os.path.basename(self._file_name)))
        self._var_recorder = CellVarRecorder(self._file_name, self._output_dir, self._variable_name, buffer_data=False)

    def initialize(self, sim):
        self._gids = list(sim.net.get_node_set(self._node_set).gids())
        self._nest_ids = [sim.net._gid2nestid[gid] for gid in self._gids]

        self._tstart = self._tstart or sim.tstart
        self._tstop = self._tstop or sim.tstop
        self._interval = self._interval or sim.dt
        self._multimeter = nest.Create('multimeter',
                                       params={'interval': self._interval, 'start': self._tstart, 'stop': self._tstop,
                                               'to_file': True, 'to_memory': False,
                                               'withtime': True,
                                               'record_from': self._variable_name,
                                               'label': self.__output_label})

        nest.Connect(self._multimeter, self._nest_ids)

    def finalize(self, sim):
        io.barrier()  # Makes sure all nodes finish, but not sure if actually required by nest

        # min_delay needs to be fetched after simulation otherwise the value will be off. There also seems to be some
        # MPI barrier inside GetKernelStatus
        self._min_delay = nest.GetKernelStatus('min_delay')
        # print self._min_delay
        if self._to_h5 and MPI_RANK == 0:
            for gid in self._gids:
                self._var_recorder.add_cell(gid, sec_list=[0], seg_list=[0.0])

            # Initialize hdf5 file including preallocated data block of recorded variables
            #   Unfortantely with NEST the final time-step recorded can't be calculated in advanced, and even with the
            # same min/max_delay can be different. We need to read the output-file to get n_steps
            def get_var_recorder(node_recording_df):
                if not self._var_recorder.is_initialized:
                    self._var_recorder.tstart = node_recording_df['time'].min()
                    self._var_recorder.tstop = node_recording_df['time'].max()
                    self._var_recorder.dt = self._interval
                    self._var_recorder.initialize(len(node_recording_df))

                return self._var_recorder

            gid_map = sim.net._nestid2gid
            for nest_file in glob.glob('{}*'.format(self.__output_label)):
                report_df = pd.read_csv(nest_file, index_col=False, names=['nest_id', 'time']+self._variable_name,
                                        sep='\t')
                for grp_id, grp_df in report_df.groupby(by='nest_id'):
                    gid = gid_map[grp_id]
                    vr = get_var_recorder(grp_df)
                    for var_name in self._variable_name:
                        vr.record_cell_block(gid, var_name, grp_df[var_name])

                if self._delete_dat:
                    # remove csv file created by nest
                    os.remove(nest_file)

            self._var_recorder.close()

        io.barrier()

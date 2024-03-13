import os
import glob
import pandas as pd
import nest

from bmtk.simulator.pointnet.io_tools import io
from bmtk.simulator.pointnet.modules.sim_module import SimulatorMod


try:
    MPI_RANK = nest.Rank()
    N_HOSTS = nest.NumProcesses()

except Exception as e:
    MPI_RANK = 0
    N_HOSTS = 1


class WeightRecorder(SimulatorMod):
    def __init__(self, name, nest_model, model_template, file_name, output_dir='.', **kwargs):
        self._name = name
        self._nest_model = nest_model
        self._model_template = model_template
        self._output_dir = output_dir

        if os.path.isabs(file_name):
            self._file_name = file_name
        else:
            abs_tmp = os.path.abspath(output_dir)
            abs_fname = os.path.abspath(file_name)
            if not abs_fname.startswith(abs_tmp):
                self._file_name = os.path.join(self._output_dir, file_name)
            else:
                self._file_name = file_name

        self._wr = None
        self._label = os.path.join(self._output_dir, self._name)        
        self._recorder_props = kwargs.get('recorder_properties', {})
        self._clean_temp_file = kwargs.get('clean_temp_file', True)

    def preload(self, sim):
        if 'label' not in self._recorder_props:
            self._recorder_props['label'] = self._label

        if 'record_to' not in self._recorder_props:
            self._recorder_props['record_to'] = 'ascii'

        self._wr = nest.Create('weight_recorder', self._recorder_props)
        nest.CopyModel(self._nest_model, self._model_template, {'weight_recorder': self._wr})       

    def finalize(self, sim):
        io.barrier()  # Makes sure all nodes finish, but not sure if actually required by nest

        if MPI_RANK == 0:
            combined_df = None            
            for nest_file in glob.glob('{}*'.format(self._recorder_props['label'])):
                report_df = pd.read_csv(nest_file, index_col=False, sep='\t', comment='#')
                node_ids, pops = sim.net.gid_map.get_node_ids(report_df['sender'].values)
                report_df['source_node_id'] = node_ids
                report_df['source_population'] = pops

                node_ids, pops = sim.net.gid_map.get_node_ids(report_df['targets'].values)
                report_df['target_node_id'] = node_ids
                report_df['target_population'] = pops

                report_df = report_df[['source_node_id', 'source_population', 'target_node_id', 'target_population', 'time_ms', 'weights', 'receptors', 'ports']]
                combined_df = report_df if combined_df is None else pd.concat([combined_df, report_df]) 

                if self._clean_temp_file:
                    os.remove(nest_file)

            combined_df.to_csv(self._file_name, sep=' ', index=False)

        io.barrier()

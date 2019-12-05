import tempfile
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class SpikesReader:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        # from bmtk.utils.reports.spike_trains import SpikeTrains
        pass

    def __read_spikes_orig(self):
        from bmtk.utils.io.spike_trains import SpikesInput
        params = {
            'input_file': '/local1/workspace/bmtk/docs/examples/NWB_files/lgn_spikes.nwb',
            'node_set': 'lgn',
            'trial': 'trial_0'
        }
        spike_trains = SpikesInput.load('LGN_spikes', 'nwb', 'spikes', params)
        for node_id in range(2000):
            node_spikes = spike_trains.get_spikes(node_id)

    def __read_spikes(self):
        from bmtk.utils.reports.spike_trains import SpikeTrains
        spike_trains = SpikeTrains.from_nwb('/local1/workspace/bmtk/docs/examples/NWB_files/lgn_spikes.nwb')
        for node_id in range(2000):
            node_spikes = spike_trains.get_times(node_id)

    def peakmem_readspikes_nwb_orig(self):
        self.__read_spikes_orig()

    def time_readspikes_nwb_orig(self):
        self.__read_spikes_orig()

    def peakmem_readspikes_nwb(self):
        self.__read_spikes()

    def time_readspikes_nwb(self):
        self.__read_spikes()


class SpikesBuilder:
    def __build_spikes(self):
        from bmtk.utils.io.spike_trains import PoissonSpikesGenerator
        tmpout = tempfile.NamedTemporaryFile(suffix='.h5')  # tempfile.mkstemp(suffix='.h5')

        psg = PoissonSpikesGenerator(range(10000), 50.0, tstop=5000.0)
        psg.to_hdf5(tmpout.name)

    '''
    def time_poission_write_orig(self):
        self.__build_spikes()

    def peakmem_poission_write_orig(self):
        self.__build_spikes()
    '''

    def __spike_writer_orig(self):
        from bmtk.utils.io.spike_trains import SpikeTrainWriter
        tmpdir = tempfile.mkdtemp()

        spike_trains = SpikeTrainWriter(tmpdir)
        for node_id in range(1000):
            spike_trains.add_spike(1.0, node_id)

        for node_id in range(1000, 3000):
            spike_trains.add_spikes(np.linspace(0, 2000.0, 1000), node_id)

        spike_trains.flush()
        spike_trains.close()

    def __spike_writer(self):
        # from bmtk.utils.reports.spike_trains import SpikeTrains
        from bmtk.utils.reports.spike_trains.spike_train_buffer import STBufferedWriter as SpikeTrains
        tmpdir = tempfile.mkdtemp()

        spike_trains = SpikeTrains(tmpdir)
        for node_id in range(1000):
            spike_trains.add_spike(node_id, 1.0)

        for node_id in range(1000, 3000):
            spike_trains.add_spikes(node_id, np.linspace(0.0, 2000.0, 1000))

        spike_trains.flush()
        # spike_trains.close()

    def time_create_spikes_file_orig(self):
        self.__spike_writer_orig()

    def peakmem_create_spikes_file_orig(self):
        self.__spike_writer_orig()

    def time_create_spikes_file(self):
        self.__spike_writer()

    def peakmem_create_spikes_file(self):
        self.__spike_writer()


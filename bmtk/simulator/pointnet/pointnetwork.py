import os
import glob
import h5py

import bmtk.simulator.pointnet.config as cfg
from bmtk.simulator.pointnet.cell import NestCell, VirtualCell
from bmtk.simulator.pointnet.property_schemas import CellTypes
import bmtk.simulator.pointnet.io as io

import nest


class PointNetwork(object):
    """Creates a network of NEST cells and connections from a graph, simulates and saves the output.

    Takes in a built PointGraph to build the network. For best results use PointNetwork.from_json(config, graph) to
    create the simulation input and parameters from a config file, the use the run() function to simulate.

    TODO:
        * Save parameters like membrane voltage using a multimeter on individual nodes.
        * Add ability to insert current and voltage clamps directly into internal nodes.
    """
    def __init__(self, graph, dt=0.001, overwrite=True):
        self._duration = 0.0  # simulation time
        self._dt = dt  # time step
        self._output_dir = './output/'  # directory where log and temporary output will be stored
        self._overwrite = overwrite
        self._block_run = False
        self._block_size = -1

        self._cells_built = False
        self._internal_connections_built = False

        self._graph = graph
        self._external_cells = {}  # dict-of-dict of external pointnet cells with keys [network_name][cell_id]
        self._internal_cells = {}  # dictionary of internal pointnet cells with cell_id as key
        self._nest_id_map = {}  # a map between NEST IDs and Node-IDs

        self._spikedetector = None
        self._spikes_file = None  # File where all output spikes will be collected and saved
        self._tmp_spikes_file = None  # temporary gdf files of spike-trains
        self._spike_trains_ds = {}  # used to temporary store NWB datasets containing spike trains

        # Reset the NEST kernel for a new simualtion
        # TODO: move this into it's own function and make sure it is called before network is built
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": self._dt, "overwrite_files": self._overwrite, "print_time": True})

    @property
    def dt(self):
        # TODO: validated dt > 0.0
        return self._dt

    @property
    def spikes_file(self):
        return self._spikes_file

    @spikes_file.setter
    def spikes_file(self, value):
        self._spikes_file = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def block_run(self):
        return self._block_run

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        self._block_run = True
        self._block_size = value

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        # TODO: validate that it is a positive number
        self._duration = value

    def build_cells(self):
        """Build NEST-based cells from graph"""
        # Create spike detector to attach to all internal nodes
        self._spikes_file = self.spikes_file  # location where spikes will be written.
        self._tmp_spikes_file = os.path.join(self.output_dir, 'tmp_spike_time')  # temporary gdf files
        self._spikedetector = nest.Create("spike_detector", 1)
        nest.SetStatus(self._spikedetector, {'label': os.path.join(self.output_dir, 'tmp_spike_times'),
                                             'withtime': True, 'withgid': True, 'to_file': True})

        # build internal nodes
        # TODO: since networks can be mixed we should loop around entire network checking if nodes are virtual or not.
        for node in self._graph.get_internal_nodes():
            ncell = NestCell(node)
            ncell.set_spike_detector(self._spikedetector)
            self._internal_cells[node.node_id] = ncell
            self._nest_id_map[ncell.nest_id] = node.node_id

        # build external nodes
        for network in self._graph.external_networks():
            self._external_cells[network] = {}
            for node in self._graph.get_nodes(network):
                if node.model_class == CellTypes.Virtual:
                    # TODO: if dynamics_params are globally set then we can create all nodes at once.
                    # TODO: Put nest_id param in PointGraph.Node and we can set vcell.nest_id =
                    vcell = VirtualCell(node)
                    self._external_cells[network][vcell.node_id] = vcell

        self._cells_built = True

    def set_recurrent_connections(self):
        """Creates recurrent (internal) connections"""
        for src_network in self._graph.internal_networks():
            for trg_gid, trg_cell in self._internal_cells.items():
                for _, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, src_network):
                    src_cell = self._internal_cells[src_prop.node_id]
                    trg_cell.set_synaptic_connection(src_cell, trg_cell, edge_prop)

    def set_external_connections(self, source_network):
        """Connect virtual nodes of an external network onto the internal network.

        :param source_network: Name of external network that targets internal nodes.
        """
        # for every internal target get source nodes and edges that connect to it create a NEST connection.
        for trg_gid, trg_cell in self._internal_cells.items():
            for _, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, source_network):
                src_cell = self._external_cells[source_network][src_prop.node_id]  # NEST implementation of source
                trg_cell.set_synaptic_connection(src_cell, trg_cell, edge_prop)

    def add_spikes_nwb(self, network, nwb_file, trial):
        """Adds spike trains from nwb file

        :param network: name of external network to add spike trains.
        :param nwb_file: NWB file with spike trains for a subset of gids in network
        :param trial: trail name in NWB file (processing/trial/spike_trains/...)
        """
        h5_file = h5py.File(nwb_file, 'r')
        self._spike_trains_ds[network] = h5_file['processing'][trial]['spike_train']

    def _get_spike_trains(self, src_gid, network):
        if network in self._spike_trains_ds:
            h5ds = self._spike_trains_ds[network]
            src_gid_str = str(src_gid)
            if src_gid_str in h5ds.keys():
                return h5ds[src_gid_str]['data']

        return None

    def make_stims(self):
        """Initialize all stimulations (spikes, injections, etc)"""
        # TODO: this is a hold-over from bionet, it may be better to set stimulations in their respective functions.
        # TODO: it is very slow, investigate
        for network in self._graph.external_networks():
            # For each external node in the graph grab the spike-trains
            # TODO: skip if external network is not connected.
            for node_id, node in self._external_cells[network].items():
                spikes = self._get_spike_trains(node_id, network)
                if spikes is not None:
                    node.set_spike_train(spikes[:])

    '''
    def __save_spike_times(self):
        # the spike detector will create many .gdf files based on nest-id and processor. Need to merge them.
        print("Saving spikes to file...")
        spikes_out = self._spikes_file
        if os.path.exists(spikes_out) and not self._overwrite:
            return

        print spikes_out
        #print spikes_file

        exit()

        gdf_files_globs = '{}/*.gdf'.format(os.path.dirname(spikes_out))
        gdf_files = glob.glob(gdf_files_globs)
        with open(spikes_out, 'w') as spikes_file:
            csv_writer = csv.writer(spikes_file, delimiter=' ')
            for gdffile in gdf_files:
                spikes_df = pd.read_csv(gdffile, names=['gid', 'time', 'nan'], sep='\t')
                for _, row in spikes_df.iterrows():
                    csv_writer.writerow([row['time'], self._nest_id_map[int(row['gid'])]])
                os.remove(gdffile)
        print("done.")
    '''

    def _get_block_trial(self, duration):
        """
        Compute necessary number of block trials, the length of block simulation and the simulation length of the last
        block run, if necessary.
        """
        if self._block_run:
            data_res = self._block_size * self._dt
            fn = duration / data_res
            n = int(fn)
            res = fn - n
        else:
            n = -1
            res = -1
            data_res = -1
        return n, res, data_res

    def run(self, duration=None):
        if duration is None:
            duration = self.duration

        n, res, data_res = self._get_block_trial(duration)
        if n > 0:
            for r in xrange(n):
                nest.Simulate(data_res)
        if res > 0:
            nest.Simulate(res * self.dt)
        if n < 0:
            nest.Simulate(duration)

        io.collect_gdf_files(self.output_dir, self._spikes_file, self._nest_id_map, self._overwrite)
        #self.__save_spike_times()

    @classmethod
    def from_json(cls, configure, graph):
        # load the json file or object
        if isinstance(configure, basestring):
            config = cfg.from_json(configure, validate=True)
        elif isinstance(configure, dict):
            config = configure
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(configure, type(configure)))

        if 'run' not in config:
            raise Exception('Json file is missing "run" entry. Unable to build Bionetwork.')
        run_dict = config['run']

        # Get network parameters
        # step time (dt) is set in the kernel and should be passed
        overwrite = run_dict['overwrite_output_dir'] if 'overwrite_output_dir' in run_dict else True
        dt = run_dict['dt']  # TODO: make sure dt exists
        network = cls(graph, dt=dt, overwrite=overwrite)

        if 'output_dir' in config['output']:
            network.output_dir = config['output']['output_dir']

        network.spikes_file = config['output']['spikes_ascii']

        if 'block_run' in run_dict and run_dict['block_run']:
            if 'block_size' not in run_dict:
                raise Exception('"block_run" is set to True but "block_size" not found.')
            network._block_size = run_dict['block_size']

        if 'duration' in run_dict:
            network.duration = run_dict['duration']

        # Create the output-directory, or delete existing files if it already exists
        io.log('setting up output directory')
        if not os.path.exists(config['output']['output_dir']):
            os.mkdir(config['output']['output_dir'])
        elif overwrite:
            for gfile in glob.glob(os.path.join(config['output']['output_dir'], '*')):
                os.remove(gfile)

        # build the cells
        io.log('building cells')
        network.build_cells()

        # Build internal connections
        if run_dict['connect_internal']:
            io.log('creating recurrent connections')
            network.set_recurrent_connections()

        # Build external connections. Set connection to default True and turn off only if explicitly stated.
        # NOTE: It might be better to set to default off?!?! Need to dicuss what would be more intuitive for the users.
        # TODO: ignore case of network name
        external_network_settings = {name: True for name in graph.external_networks()}
        if 'connect_external' in run_dict:
            external_network_settings.update(run_dict['connect_external'])
        for netname, connect in external_network_settings.items():
            if connect:
                io.log('Setting external connections for {}'.format(netname))
                network.set_external_connections(netname)

        # Build inputs
        if 'input' in config:
            for netinput in config['input']:
                if netinput['type'] == 'external_spikes' and netinput['format'] == 'nwb' and netinput['active']:
                    network.add_spikes_nwb(netinput['network'], netinput['file'], netinput['trial'])

            io.log('Adding stimulations')
            network.make_stims()

        return network

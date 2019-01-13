import csv

from bmtk.simulator.core.simulator import Simulator
import bmtk.simulator.utils.simulation_inputs as inputs
from bmtk.simulator.filternet.config import Config
from bmtk.simulator.filternet.lgnmodel.movie import *
from bmtk.simulator.filternet import modules as mods
from bmtk.simulator.filternet.io_tools import io
from six import string_types


class FilterSimulator(Simulator):
    def __init__(self, network, dt, tstop):
        super(FilterSimulator, self).__init__()
        self._network = network
        self._dt = dt
        self._tstop = tstop/1000.0

        self.rates_csv = None
        self._movies = []

    def add_movie(self, movie_type, params):
        # TODO: Move this into its own factory
        movie_type = movie_type.lower() if isinstance(movie_type, string_types) else 'movie'
        if movie_type == 'movie' or not movie_type:
            raise NotImplementedError

        elif movie_type == 'full_field':
            raise NotImplementedError

        elif movie_type == 'full_field_flash':
            raise NotImplementedError

        elif movie_type == 'graiting':
            init_params = FilterSimulator.find_params(['row_size', 'col_size', 'frame_rate'], **params)
            create_params = FilterSimulator.find_params(['gray_screen_dur', 'cpd', 'temporal_f', 'theta', 'contrast'],
                                                        **params)
            gm = GratingMovie(**init_params)
            graiting_movie = gm.create_movie(t_min=0.0, t_max=self._tstop, **create_params)
            self._movies.append(graiting_movie)

        else:
            raise Exception('Unknown movie type {}'.format(movie_type))

    def run(self):
        for mod in self._sim_mods:
            mod.initialize(self)

        io.log_info('Evaluating rates.')
        for cell in self._network.cells():
            for movie in self._movies:
                ts, f_rates = cell.lgn_cell_obj.evaluate(movie, downsample=1, separable=True)

                for mod in self._sim_mods:
                    mod.save(self, cell.gid, ts, f_rates)

                """
                if self.rates_csv is not None:
                    print 'saving {}'.format(cell.gid)
                    for t, f in zip(t, f_tot):
                        csv_writer.writerow([t, f, cell.gid])
                    csv_fhandle.flush()
                """
        io.log_info('Done.')
        for mod in self._sim_mods:
            mod.finalize(self)

    """
    def generate_spikes(LGN, trials, duration, output_file_name):
        # f_tot = np.loadtxt(output_file_name + "_f_tot.csv", delimiter=" ")
        # t = f_tot[0, :]

        f = h5.File(output_file_name + "_f_tot.h5", 'r')
        f_tot = np.array(f.get('firing_rates_Hz'))

        t = np.array(f.get('time'))
        # For h5 files that don't have time explicitly saved
        t = np.linspace(0, duration, f_tot.shape[1])


        #create output file
        f = nwb.create_blank_file(output_file_name + '_spikes.nwb', force=True)

        for trial in range(trials):
            for counter in range(len(LGN.nodes())):
                try:
                    spike_train = np.array(f_rate_to_spike_train(t*1000., f_tot[counter, :], np.random.randint(10000), 1000.*min(t), 1000.*max(t), 0.1))
                except:
                    spike_train = 1000.*np.array(pg.generate_inhomogenous_poisson(t, f_tot[counter, :], seed=np.random.randint(10000))) #convert to milliseconds and hence the multiplication by 1000

                nwb.SpikeTrain(spike_train, unit='millisecond').add_to_processing(f, 'trial_%s' % trial)
        f.close()

    """


    @staticmethod
    def find_params(param_names, **kwargs):
        ret_dict = {}
        for pn in param_names:
            if pn in kwargs:
                ret_dict[pn] = kwargs[pn]

        return ret_dict

    @classmethod
    def from_config(cls, config, network):
        if not isinstance(config, Config):
            try:
                config = Config.load(config, False)
            except Exception as e:
                network.io.log_exception('Could not convert {} (type "{}") to json.'.format(config, type(config)))

        if not config.with_networks:
            network.io.log_exception('Could not find any network files. Unable to build network.')

        sim = cls(network=network, dt=config.dt, tstop=config.tstop)

        if config.jitter is not None:
            network.jitter = config.jitter

        network.io.log_info('Building cells.')
        network.build_nodes()

        # TODO: Need to create a gid selector
        for sim_input in inputs.from_config(config):
            if sim_input.input_type == 'movie':
                sim.add_movie(sim_input.module, sim_input.params)
            else:
                raise Exception('Unable to load input type {}'.format(sim_input.input_type))


            """
            node_set = network.get_node_set(sim_input.node_set)
            if sim_input.input_type == 'spikes':
                spikes = spike_trains.SpikesInput.load(name=sim_input.name, module=sim_input.module,
                                                       input_type=sim_input.input_type, params=sim_input.params)
                io.log_info('Build virtual cell stimulations for {}'.format(sim_input.name))
                network.add_spike_trains(spikes, node_set)

            elif sim_input.module == 'IClamp':
                # TODO: Parse from csv file
                amplitude = sim_input.params['amp']
                delay = sim_input.params['delay']
                duration = sim_input.params['duration']
                gids = sim_input.params['node_set']
                sim.attach_current_clamp(amplitude, delay, duration, node_set)

            elif sim_input.module == 'xstim':
                sim.add_mod(mods.XStimMod(**sim_input.params))

            else:
                io.log_exception('Can not parse input format {}'.format(sim_input.name))
            """


        rates_csv = config.output.get('rates_csv', None)
        rates_h5 = config.output.get('rates_h5', None)
        if rates_csv or rates_h5:
            sim.add_mod(mods.RecordRates(rates_csv, rates_h5, config.output_dir))

        spikes_csv = config.output.get('spikes_csv', None)
        spikes_h5 = config.output.get('spikes_h5', None)
        spikes_nwb = config.output.get('spikes_nwb', None)
        if spikes_csv or spikes_h5 or spikes_nwb:
            sim.add_mod(mods.SpikesGenerator(spikes_csv, spikes_h5, spikes_nwb, config.output_dir))

        # Parse the "reports" section of the config and load an associated output module for each report
        """
        sim_reports = reports.from_config(config)
        for report in sim_reports:
            if isinstance(report, reports.SpikesReport):
                mod = mods.SpikesMod(**report.params)

            elif isinstance(report, reports.MembraneReport):
                if report.params['sections'] == 'soma':
                    mod = mods.SomaReport(**report.params)

                else:
                    #print report.params
                    mod = mods.MembraneReport(**report.params)

            elif isinstance(report, reports.ECPReport):
                mod = mods.EcpMod(**report.params)
                # Set up the ability for ecp on all relevant cells
                # TODO: According to spec we need to allow a different subset other than only biophysical cells
                for gid, cell in network.cell_type_maps('biophysical').items():
                    cell.setup_ecp()
            else:
                # TODO: Allow users to register customized modules using pymodules
                io.log_warning('Unrecognized module {}, skipping.'.format(report.module))
                continue

            sim.add_mod(mod)
        """
        return sim
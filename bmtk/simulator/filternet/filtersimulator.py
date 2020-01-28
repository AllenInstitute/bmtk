import csv
import numpy as np

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
        self._eval_options = []

    def add_movie(self, movie_type, params):
        # TODO: Move this into its own factory
        movie_type = movie_type.lower() if isinstance(movie_type, string_types) else 'movie'
        if movie_type == 'movie' or not movie_type:
            if 'data_file' in params:
                m_data = None
                if 'data_file' in params:
                    m_data = np.load(params['data_file'])
                elif 'data' in params:
                    m_data = params['data']
                else:
                    raise Exception('Could not find movie "data_file" in config to use as input.')

                init_params = FilterSimulator.find_params(['row_range', 'col_range', 'labels', 'units', 'frame_rate',
                                                           't_range'], **params)
                self._movies.append(Movie(m_data, **init_params))

        elif movie_type == 'full_field':
            raise NotImplementedError

        elif movie_type == 'full_field_flash':
            init_params = FilterSimulator.find_params(['row_size', 'col_size', 't_on', 't_off', 'max_intensity',
                                                       'frame_rate'], **params)
            init_params['row_range'] = range(init_params['row_size'])
            del init_params['row_size']
            init_params['col_range'] = range(init_params['col_size'])
            del init_params['col_size']
            init_params['t_on'] = init_params['t_on']/1000.0
            init_params['t_off'] = init_params['t_off']/1000.0

            ffm = FullFieldFlashMovie(**init_params)
            mv = ffm.full(t_max=self._tstop)
            self._movies.append(mv)

        elif movie_type == 'graiting':
            init_params = FilterSimulator.find_params(['row_size', 'col_size', 'frame_rate'], **params)
            create_params = FilterSimulator.find_params(['gray_screen_dur', 'cpd', 'temporal_f', 'theta', 'contrast'],
                                                        **params)
            gm = GratingMovie(**init_params)
            graiting_movie = gm.create_movie(t_min=0.0, t_max=self._tstop, **create_params)
            self._movies.append(graiting_movie)

        elif movie_type == 'looming':
            init_params = FilterSimulator.find_params(['row_size', 'col_size', 'frame_rate'], **params)
            movie_params = FilterSimulator.find_params(['t_looming', 'gray_sceen_dur'], **params)
            lm = LoomingMovie(**init_params)
            looming_movie = lm.create_movie(**movie_params)
            self._movies.append(looming_movie)

        else:
            raise Exception('Unknown movie type {}'.format(movie_type))

        if 'evaluation_options' in params:
            self._eval_options.append(params['evaluation_options'])
        else:
            self._eval_options.append({})

    def run(self):
        for mod in self._sim_mods:
            mod.initialize(self)

        io.log_info('Evaluating rates.')
        for cell in self._network.cells():
            for movie, options in zip(self._movies, self._eval_options):
                ts, f_rates = cell.lgn_cell_obj.evaluate(movie, **options)

                for mod in self._sim_mods:
                    mod.save(self, cell, ts, f_rates)

        io.log_info('Done.')
        for mod in self._sim_mods:
            mod.finalize(self)

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

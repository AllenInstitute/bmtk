import csv
import numpy as np
from six import string_types

from bmtk.simulator.core.simulator import Simulator
import bmtk.simulator.utils.simulation_inputs as inputs
from bmtk.simulator.filternet.config import Config
from bmtk.simulator.filternet.lgnmodel.movie import *
from bmtk.simulator.filternet import modules as mods
from bmtk.simulator.filternet.io_tools import io
from bmtk.utils.io.ioutils import bmtk_world_comm
from bmtk.simulator.filternet.auditory_processing import AuditoryInput
import scipy.io as syio
import os


class FilterSimulator(Simulator):
    def __init__(self, network, dt, tstop):
        super(FilterSimulator, self).__init__()
        self._network = network
        self._dt = dt
        self._tstop = tstop/1000.0
        self._io = network.io

        self.rates_csv = None
        self._movies = []
        self._eval_options = []

    @property
    def io(self):
        return self._io

    @property
    def dt(self):
        return self._dt

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

                contrast_min, contrast_max = m_data.min(), m_data.max()
                normalize_data = params.get('normalize', False)
                if contrast_min < -1.0 or contrast_max > 1.0:
                    if normalize_data:
                        self.io.log_info('Normalizing movie data to (-1.0, 1.0).')
                        m_data = m_data*2.0/(contrast_max - contrast_min) - 1.0
                    else:
                        self.io.log_info('Movie data range ifind_paramss not normalized to (-1.0, 1.0).')

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
            init_params['max_intensity'] = init_params.get('max_intensity', 1)# *-1.0

            ffm = FullFieldFlashMovie(**init_params)
            mv = ffm.full(t_max=self._tstop)
            self._movies.append(mv)

        elif movie_type == 'graiting':
            init_params = FilterSimulator.find_params(['row_size', 'col_size', 'frame_rate'], **params)
            create_params = FilterSimulator.find_params(['gray_screen_dur', 'cpd', 'temporal_f', 'theta', 'contrast'],
                                                        **params)

            create_params['gray_screen_dur'] /= 1000.0
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

    def add_audio(self, audio_type, params):
        # Create cochleagram "movie" from audio wav file
        audio_type = audio_type.lower() if isinstance(audio_type, string_types) else 'movie'
        if audio_type in ['wav_file', 'mat_file'] or not audio_type:
            if 'data_file' in params:
                aud_file = params['data_file']
                if audio_type == 'mat_file':
                    n = params['stim_number']
                    wav_file = os.path.splitext(aud_file)[0] + str(n) + '.wav'
                    if not os.path.exists(wav_file):
                        mat = syio.loadmat(params['data_file'])
                        data = np.squeeze(mat['timit_sents'][0, n])
                        sr = mat['aud_fs'][0][0]
                        scaled = np.int16(data / np.max(np.abs(data)) * 32768)
                        syio.wavfile.write(wav_file, sr, scaled)
                    else:
                        io.log_warning('Wav file already exists, please delete to overwrite.')
                    aud_file = wav_file

                #elif 'data' in params:
                #    m_data = params['data']
            else:
                raise Exception('Could not find audio "data_file" in config to use as input.')

            aud = AuditoryInput(aud_file)

            #if params.get('frame_rate'):
            #    frame_rate = params.get('frame_rate')
            #else:
            init_params = FilterSimulator.find_params(['row_range', 'col_range', 'labels', 'units', 'frame_rate',
                                                       't_range', 'padding'], **params)
            if 'frame_rate' in init_params.keys():
                frame_rate = init_params['frame_rate']
            else:
                frame_rate = 1000

            coch, center_freqs_log, times = aud.get_cochleagram(frame_rate, interp_to_freq=params['interp_to_freq'])
            coch = coch.T
            #coch = np.log(coch)

            normalize_data = params.get('normalize', None)
            if normalize_data == 'full' or normalize_data == True:
                contrast_min, contrast_max = coch.min(), coch.max()
                self.io.log_info('Normalizing auditory input to (-1.0, 1.0).')
                coch = (coch-contrast_min)*2.0/(contrast_max - contrast_min) - 1.0
            elif normalize_data == 'relative':
                self.io.log_info('Auditory input is normalized maintaining relative amplitude')
                coch = coch*2.8
            else:
                self.io.log_info('Auditory input range is not normalized.')

            coch = coch[:,:, np.newaxis]

            # Note, overwrites these if user supplied, instead taken from cochleagram
            init_params['row_range'] = center_freqs_log
            init_params['col_range'] = [0]
            init_params['t_range'] = times
            #? Frame_rate
            # Dimensions of time, row, column
            self._movies.append(Movie(coch, **init_params))
        else:
            raise Exception('Unknown audio type {}'.format(audio_type))

        if 'evaluation_options' in params:
            self._eval_options.append(params['evaluation_options'])
        else:
            self._eval_options.append({})

    def run(self):
        for mod in self._sim_mods:
            mod.initialize(self)

        io.log_info('Evaluating rates.')

        cells_on_rank = self.local_cells()
        n_cells_on_rank = len(cells_on_rank)
        ten_percent = int(np.ceil(n_cells_on_rank*0.1))
        rank_msg = '' if bmtk_world_comm.MPI_size < 2 else ' (on rank {})'.format(bmtk_world_comm.MPI_rank)

        max_fr = np.empty(len(cells_on_rank))
        for cell_num, cell in enumerate(cells_on_rank):
            for movie, options in zip(self._movies, self._eval_options):
                if cell_num > 0 and cell_num % ten_percent == 0:
                    io.log_debug(' Processing cell {} of {}{}.'.format(cell_num, n_cells_on_rank, rank_msg))
                ts, f_rates = cell.lgn_cell_obj.evaluate(movie, **options)
                max_fr[cell_num] = np.max(f_rates)
                if movie.padding:
                    f_rates = f_rates[int(movie.data.shape[0]-movie.data_orig.shape[0]):]
                    ts = ts[int(movie.data.shape[0]-movie.data_orig.shape[0]):]
                    ts = ts-ts[0]

                for mod in self._sim_mods:
                    mod.save(self, cell, ts, f_rates)
        io.log_info('Max firing rate: {}'.format(np.max(max_fr)))
        io.log_info('Done.')
        for mod in self._sim_mods:
            mod.finalize(self)

    def local_cells(self):
        return self._network.cells()

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

        for sim_input in inputs.from_config(config):
            if sim_input.input_type == 'movie':
                sim.add_movie(sim_input.module, sim_input.params)
            elif sim_input.input_type == 'audio':
                sim.add_audio(sim_input.module, sim_input.params)
            else:
                raise Exception('Unable to load input type {}'.format(sim_input.input_type))

        network.io.log_info('Building cells.')
        network.build_nodes()

        rates_csv = config.output.get('rates_csv', None)
        rates_h5 = config.output.get('rates_h5', None)
        if rates_csv or rates_h5:
            sim.add_mod(mods.RecordRates(rates_csv, rates_h5, config.output_dir))

        spikes_csv = config.output.get('spikes_csv', None) or config.output.get('spikes_file_csv', None)
        spikes_h5 = config.output.get('spikes_h5', None) or config.output.get('spikes_file', None)
        spikes_nwb = config.output.get('spikes_nwb', None) or config.output.get('spikes_file_nwb', None)
        if spikes_csv or spikes_h5 or spikes_nwb:
            sim.add_mod(mods.SpikesGenerator(spikes_csv, spikes_h5, spikes_nwb, config.output_dir))

        return sim

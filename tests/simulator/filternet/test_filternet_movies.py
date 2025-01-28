from pathlib import Path
import numpy as np
from bmtk.simulator.filternet import FilterSimulator, FilterNetwork, Config
from bmtk.simulator.filternet.lgnmodel.movie import GratingMovie
from bmtk.utils.sim_setup import build_env_filternet


def test_filtersimulator_add_movie_with_phase(tmp_path):
    simulator = build_simulator(tmp_path)
    simulator.add_movie(
        'graiting',  # this is typo, but keeping it this line for backward compatibility
        {'row_size': 100,
         'col_size': 100,
         'gray_screen_dur': 0,
         'phase': 180})
    assert np.allclose(
        simulator._movies[0].data,
        GratingMovie(100, 100).create_movie(phase=180, t_max=simulator._tstop).data)

def test_filtersimulator_add_movie(tmp_path):
    simulator = build_simulator(tmp_path)
    theta = 9.59
    simulator.add_movie(
        'grating',
        {'row_size': 120,
         'col_size': 80,
         'theta': theta,
         'gray_screen_dur': 0,
         'y_dir': 'up'}
    )
    
    sample_movie = GratingMovie(120, 80, y_dir='down').create_movie(
        theta=theta,
        t_max=simulator._tstop
    )

    # When y_dir is different, the movie should be different
    assert not np.allclose(
        simulator._movies[0].data,
        sample_movie.data,
    )

    # When they match, they should be the same.
    assert np.allclose(
        simulator._movies[0].data,
        GratingMovie(120, 80, y_dir='up').create_movie(
            theta=theta,
            t_max=simulator._tstop
        ).data,
    )

    # test flip_y
    simulator.add_movie(
        'movie',
        {'data': sample_movie.data,
         'frame_rate': 1000.0,
         'flip_y': True,
         'y_dir': 'down'}
    )

    # flipped movie should have -theta direction.
    # This won't be an exact match, because these two movies' starting point of the
    # phase is different, but by adjusting theta to 9.59, we can make them have one
    # cycle vertically, and they become close.
    target_movie = GratingMovie(120, 80, y_dir='down').create_movie(
        theta=-theta,
        t_max=simulator._tstop
    )
    
    assert np.allclose(
        simulator._movies[1].data,
        target_movie.data,
        atol=1e-2
    )
    

def build_simulator(tmp_path):
    data_dir = Path(__file__).parent / 'data' 
    build_env_filternet(
        base_dir=tmp_path,
        network_dir=str(data_dir / "network"),
        tstop=100,
        include_examples=True,
        config_file=tmp_path / 'config.json')

    config = Config.from_json(str(tmp_path / 'config.json'))
    config.build_env()
    net = FilterNetwork.from_config(config)
    sim = FilterSimulator.from_config(config, net)
    return sim

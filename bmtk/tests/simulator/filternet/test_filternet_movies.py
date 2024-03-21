from pathlib import Path
import numpy as np
from bmtk.simulator.filternet import FilterSimulator, FilterNetwork, Config
from bmtk.simulator.filternet.lgnmodel.movie import GratingMovie
from bmtk.utils.sim_setup import build_env_filternet


def test_filtersimulator_add_movie_with_phase(tmp_path):
    simulator = build_simulator(tmp_path)
    simulator.add_movie(
        'graiting',
        {'row_size': 100,
         'col_size': 100,
         'gray_screen_dur': 0,
         'phase': 180})
    assert np.allclose(
        simulator._movies[0].data,
        GratingMovie(100, 100).create_movie(phase=180, t_max=simulator._tstop).data)


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

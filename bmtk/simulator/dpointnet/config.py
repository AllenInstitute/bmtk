from bmtk.simulator.core.simulation_config import SimulationConfig
from .io_tools import io


class Config(SimulationConfig):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.io = io
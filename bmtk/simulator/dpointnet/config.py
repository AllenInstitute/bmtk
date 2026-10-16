from collections.abc import Mapping
import os

from bmtk.simulator.core.simulation_config import SimulationConfig
from .io_tools import io


class Config(SimulationConfig):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.io = io

    def build_env(self, force=False):
        distribution = self.run.get("distribution")
        strategy = distribution.get("strategy") if isinstance(distribution, Mapping) else None
        launched = os.environ.get("BMTK_DPOINTNET_MPI_LAUNCH") in ("cpu", "gpu")
        if strategy == "mpi" and not launched:
            raise ValueError(
                "MPI training requires mpiexec -n N python -m "
                "bmtk.utils.dpointnet_mpi training_script.py [arguments]."
            )
        if launched and strategy != "mpi":
            raise ValueError(
                'The MPI launcher requires run.distribution.strategy="mpi" '
                "in the configuration."
            )
        self.io.configure_mpi()
        super().build_env(force=force)

    def copy_to_output(self):
        if self.io.mpi_rank == 0:
            super().copy_to_output()
        self.io.barrier()
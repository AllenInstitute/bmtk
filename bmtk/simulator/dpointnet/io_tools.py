import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime
from threading import Lock

from bmtk.simulator.core.io_tools import IOUtils
from bmtk.utils.io.ioutils import bmtk_world_comm


class RNNIOUtils(IOUtils):
    _logger = None

    def __init__(self):
        super(RNNIOUtils, self).__init__()
        self._diagnostic_path = None
        self._diagnostic_lock = Lock()
        self._console_handler = None
        self.configure_mpi()

    def configure_mpi(self):
        mode = os.environ.get("BMTK_DPOINTNET_MPI_LAUNCH")
        if mode is not None and mode not in ("cpu", "gpu"):
            raise ValueError("Unknown DPointNet MPI launcher mode.")
        self._comm = bmtk_world_comm.comm if mode is not None else None
        if mode is not None and self._comm is None:
            raise RuntimeError("DPointNet MPI launch requires an MPI communicator.")
        self.mpi_rank = self._comm.Get_rank() if self._comm is not None else 0
        self.mpi_size = self._comm.Get_size() if self._comm is not None else 1

    def barrier(self):
        if self._comm is not None:
            self._comm.Barrier()

    def setup_output_dir(self, output_dir, log_file, overwrite=True):
        super().setup_output_dir(
            output_dir, log_file if self.mpi_rank == 0 else None, overwrite
        )
        self.barrier()

    def log_exception(self, message, raise_exception=True):
        if self._comm is None:
            return super().log_exception(message, raise_exception=raise_exception)
        if self.mpi_rank == 0:
            self.logger.error(message)
        # Let rank-local failures reach the launcher's abort without a collective.
        if raise_exception:
            raise Exception(message)

    @IOUtils.log_to_console.setter
    def log_to_console(self, flag):
        IOUtils.log_to_console.fset(self, flag)
        if RNNIOUtils._logger is not None:
            if flag:
                self._set_console_logging()
            elif self._console_handler is not None:
                self.logger.removeHandler(self._console_handler)
                self._console_handler.close()
                self._console_handler = None

    def _set_console_logging(self):
        if self._log_to_console and self._console_handler is None:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setFormatter(self._log_format)
            self._logger.addHandler(self._console_handler)

    def set_log_level(self, loglevel):
        super(RNNIOUtils, self).set_log_level(loglevel)
        if RNNIOUtils._logger is not None:
            self.logger.setLevel(self._log_level)

    def set_log_format(self, format_str):
        super(RNNIOUtils, self).set_log_format(format_str)
        if self._console_handler is not None:
            self._console_handler.setFormatter(self._log_format)

    def save_exception(self, exception, context):
        """Save recovery details without sending a traceback to console handlers."""
        with self._diagnostic_lock:
            try:
                if self._diagnostic_path is None:
                    directory = next(
                        (
                            os.path.dirname(handler.baseFilename)
                            for handler in self.logger.handlers
                            if isinstance(handler, logging.FileHandler)
                        ),
                        None,
                    )
                    with tempfile.NamedTemporaryFile(
                        mode="w", prefix="dpointnet-diagnostics-",
                        suffix=".log", dir=directory, delete=False,
                    ) as stream:
                        self._diagnostic_path = stream.name
                with open(self._diagnostic_path, "a") as stream:
                    stream.write(f"{datetime.now().isoformat()} {context}\n")
                    stream.writelines(traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    ))
                    stream.write("\n")
                return self._diagnostic_path
            except OSError as error:
                self.log_warning(
                    f"Unable to save DPointNet recovery diagnostics: {error}. "
                    f"Original error: {type(exception).__name__}: {exception}"
                )
                return None

    @property
    def logger(self):
        if RNNIOUtils._logger is None:
            RNNIOUtils._logger = logging.getLogger(__name__)
            RNNIOUtils._logger.setLevel(self._log_level)
            RNNIOUtils._logger.propagate = False
            if not RNNIOUtils._logger.handlers:
                self._set_console_logging()
        return RNNIOUtils._logger


io = RNNIOUtils()

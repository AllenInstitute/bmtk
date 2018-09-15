import os
import sys
import shutil
import logging


class IOUtils(object):
    """
    For logging/mkdir commands we sometimes need to use different MPI classes depending on the simulator being used
    (NEST and NEURON have their own barrier functions that don't work well with mpi). We also need to be able to
    adjust the logging levels/format at run-time depending on the simulator/configuration options.

    Thus the bulk of the io and logging functions are put into their own class and can be overwritten by specific
    simulator modules
    """
    def __init__(self):
        self.mpi_rank = 0
        self.mpi_size = 1

        self._log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        self._log_level = logging.DEBUG
        self._log_to_console = True
        self._logger = None

    @property
    def log_to_console(self):
        return self._log_to_console

    @log_to_console.setter
    def log_to_console(self, flag):
        assert(isinstance(flag, bool))
        self._log_to_console = flag

    @property
    def logger(self):
        if self._logger is None:
            # Create the logger the first time it is accessed
            self._logger = logging.getLogger(self.__class__.__name__)
            self._logger.setLevel(self._log_level)
            self._set_console_logging()

        return self._logger

    def _set_console_logging(self):
        if not self._log_to_console:
            return

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._log_format)
        self._logger.addHandler(console_handler)

    def set_log_format(self, format_str):
        self._log_format = logging.Formatter(format_str)

    def set_log_level(self, loglevel):
        if isinstance(loglevel, int):
            self._log_level = loglevel

        elif isinstance(loglevel, (str, unicode)):
            self._log_level = logging.getLevelName(loglevel)

        else:
            raise Exception('Error: cannot set logging levels to {}'.format(loglevel))

    def barrier(self):
        pass

    def quiet_simulator(self):
        pass

    def setup_output_dir(self, output_dir, log_file, overwrite=True):
        if self.mpi_rank == 0:
            # Create output directory
            if os.path.exists(output_dir):
                if overwrite:
                    shutil.rmtree(output_dir)
                else:
                    self.log_exception('Directory already exists (remove or set to overwrite).')
            os.makedirs(output_dir)

            # Create log file
            if log_file is not None:
                log_path = log_file if os.path.isabs(log_file) else os.path.join(output_dir, log_file)
                file_logger = logging.FileHandler(log_path)
                file_logger.setFormatter(self._log_format)
                self.logger.addHandler(file_logger)
                self.log_info('Created log file')

        self.barrier()

    def log_info(self, message, all_ranks=False):
        if all_ranks is False and self.mpi_rank != 0:
            return

        self.logger.info(message)

    def log_warning(self, message, all_ranks=False):
        if all_ranks is False and self.mpi_rank != 0:
            return

        self.logger.warning(message)

    def log_exception(self, message):
        if self.mpi_rank == 0:
            self.logger.error(message)

        self.barrier()
        raise Exception(message)


io = IOUtils()
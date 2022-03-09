import os
import sys
import shutil
import logging
from six import text_type


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
        self._log_level = logging.INFO
        self._log_to_console = True
        self._logger = None

        # Used by log_warning to keep track of previous shown messages
        self._warn_messages = set()

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

        elif isinstance(loglevel, (str, text_type)):
            self._log_level = logging.getLevelName(loglevel)

        else:
            raise Exception('Error: cannot set logging levels to {}'.format(loglevel))

    def barrier(self):
        """MPI Barrier call"""
        pass  # By default this does nothing, if a simulator is to implement mpi support it should overwrite.

    def quiet_simulator(self):
        """Turns off logging/messages of the native simulator"""
        pass  # Simulators should implement their own versions

    def setup_output_dir(self, output_dir, log_file, overwrite=True):
        if self.mpi_rank == 0:
            # Create output directory, do it only on one rank to prevent overwrite errors
            if os.path.exists(output_dir):
                if overwrite:
                    shutil.rmtree(output_dir)
                    os.makedirs(output_dir)
                else:
                    self.log_warning('Directory {} already exists.'.format(output_dir))
            else:
                os.makedirs(output_dir)
        self.barrier()  # other ranks wait for output directory to be created.

        # Create logger handle for writing to log.txt file
        if log_file is not None:
            file_logger = logging.FileHandler(log_file)
            file_logger.setFormatter(self._log_format)
            self.logger.addHandler(file_logger)
            self.log_info('Created log file', all_ranks=False)  # write first message only on rank 0

    def log_info(self, message, all_ranks=False):
        if all_ranks is False and self.mpi_rank != 0:
            return

        self.logger.info(message)

    def log_warning(self, message, all_ranks=False, display_once=False):
        if all_ranks is False and self.mpi_rank != 0:
            return

        # Prevent the spaming of the same warning message multiple times, keep track of previous messages and if
        # display_once is true don't log.
        # TODO: Won't work across multiple ranks in MPI, is it possible?
        msg_hash = hash(message)
        if display_once and msg_hash in self._warn_messages:
            return
        self._warn_messages.add(msg_hash)

        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def log_exception(self, message, raise_exception=True):
        if self.mpi_rank == 0:
            self.logger.error(message)

        self.barrier()
        if raise_exception:
            raise Exception(message)


io = IOUtils()
import os
import shutil
import logging


# TODO: Need more generic logging for
class IOUtils(object):
    def __init__(self):
        self.mpi_rank = 0
        self.mpi_size = 1

        self._log_format = '%(asctime)s [%(levelname)s] %(message)s'
        self._logger = logging.getLogger()
        self.set_console_logging()

    @property
    def logger(self):
        return None

    def set_console_logging(self):
        pass

    def barrier(self):
        pass

    def quit(self):
        exit(1)

    def setup_output_dir(self, config_dir, log_file, overwrite=True):
        if self.mpi_rank == 0:
            # Create output directory
            if os.path.exists(config_dir):
                if overwrite:
                    shutil.rmtree(config_dir)
                else:
                    self.log_exception('ERROR: Directory already exists (remove or set to overwrite).')
            os.makedirs(config_dir)

            # Create log file
            if log_file is not None:
                file_logger = logging.FileHandler(log_file)
                file_logger.setFormatter(self._log_format)
                self.logger.addHandler(file_logger)
                self.log_info('Created log file')

        self.barrier()

    def log_info(self, message, all_ranks=False):
        print(message)

    def log_warning(self, message, all_ranks=False):
        print(message)

    def log_exception(self, message):
        raise Exception(message)

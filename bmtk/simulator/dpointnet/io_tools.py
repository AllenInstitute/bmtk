import logging

from bmtk.simulator.core.io_tools import IOUtils


class RNNIOUtils(IOUtils):
    _logger = None

    def __init__(self):
        super(RNNIOUtils, self).__init__()

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

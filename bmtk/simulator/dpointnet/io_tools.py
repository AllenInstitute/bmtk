import logging
import os
import tempfile
import traceback
from datetime import datetime
from threading import Lock

from bmtk.simulator.core.io_tools import IOUtils


class RNNIOUtils(IOUtils):
    _logger = None

    def __init__(self):
        super(RNNIOUtils, self).__init__()
        self._diagnostic_path = None
        self._diagnostic_lock = Lock()

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

# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import json
import warnings

from bmtk.utils.sonata.config import SonataConfig, copy_config
from bmtk.simulator.core.io_tools import io


class SimulationConfig(SonataConfig):
    """A special version of SonataConfig that contains some methods that can be used by the simulators. Mainly
    it contains the build_env() method which can be called to setup logging plus initialize the 'output_dir' folder.

    Functionality that is specific to SONATA should go in SonataConfig
    """

    def __init__(self, *args, **kwargs):
        super(SimulationConfig, self).__init__(*args, **kwargs)
        self.env_built = False
        self._io = None

    @property
    def io(self):
        if self._io is None:
            self._io = io
        return self._io

    @io.setter
    def io(self, io):
        self._io = io

    @property
    def validator(self):
        if self._validator is None:
            from .simulation_config_validator import SimulationConfigValidator

            json_schema = os.path.join(os.path.dirname(__file__), 'sonata_schemas', 'config_schema.json')
            with open(json_schema, 'r') as f:
                config_schema = json.load(f)
                self._validator = SimulationConfigValidator(schema=config_schema)

        return self._validator

    def validate(self):
        try:
            return super(SimulationConfig, self).validate()
        except Exception as exc:
            # Capture the output into our log file before raising the error
            msg = 'SimulationConfig ValidationError: {}'.format(str(exc))
            self.io.log_exception(msg)

    def copy_to_output(self):
        copy_config(self)

    def _set_logging(self):
        """Check if log-level and/or log-format string is being changed through the config"""
        output_sec = self.output
        if 'log_format' in output_sec:
            self.io.set_log_format(output_sec['log_format'])

        if 'log_level' in output_sec:
            self.io.set_log_level(output_sec['log_level'])

        if 'log_to_console' in output_sec:
            self.io.log_to_console = output_sec['log_to_console']

        if 'quiet_simulator' in output_sec and output_sec['quiet_simulator']:
            self.io.quiet_simulator()

    def build_env(self, force=False):
        """Creates the folder(s) set in 'output' section, sets up logging and copies over the configuration"""
        if self.env_built and not force:
            return

        self._set_logging()
        self.io.setup_output_dir(self.output_dir, self.log_file, self.overwrite_output)
        self.copy_to_output()
        self.env_built = True


def from_dict(config_dict, validator=None, **opts):
    warnings.warn('Deprecated: Pleas use SimulationConfig.from_dict() instead.', DeprecationWarning)
    return SimulationConfig.from_dict(config_dict, validator, **opts)


def from_json(config_file, validator=None, **opts):
    warnings.warn('Deprecated: Pleas use SimulationConfig.from_json() instead.', DeprecationWarning)
    return SimulationConfig.from_json(config_file, validator, **opts)

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
from six import string_types

from bmtk.simulator.core.simulation_config import SimulationConfig
from bmtk.simulator.core.io_tools import io
from .config import Config
from .ssn import inputs_generator, init_function, activation_function


class PopNetwork:
    @staticmethod
    def from_config(conf, **properties):
        if isinstance(conf, SimulationConfig):
            config = conf
        else:
            try:
                config = SimulationConfig.load(conf)
            except Exception as e:
                io.log_exception('Could not convert {} (type "{}") to json.'.format(conf, type(conf)))

        if config.target_simulator is None:
            io.log_debug('Unspecified PopNet "target_simulator", defaulting to SSN (Options: SSN, DiPDE)')
        elif config.target_simulator.lower() == 'dipde':
            from .dipde import PopNetwork
            return PopNetwork.from_config(config, **properties)
        elif config.target_simulator.lower() == 'ssn':
            from .ssn import PopNetwork
            return PopNetwork.from_config(config, **properties)
        else:
            io.log_exception(f'Unrecognized PopNet target_simulator "{config.target_simulator}')


class PopSimulator:
    @staticmethod
    def from_config(configure, network, **properties):
        if network.target_simulator is None:
            io.log_debug('Unspecified PopNet "target_simulator", defaulting to SSN (Options: SSN, DiPDE)')
        elif network.target_simulator.lower() == 'dipde':
            from .dipde import PopSimulator
            return PopSimulator.from_config(configure, network, **properties)
        elif network.target_simulator.lower() == 'ssn':
            from .ssn import PopSimulator
            return PopSimulator.from_config(configure, network, **properties)
        else:
            io.log_exception(f'Unrecognized PopNet target_simulator "{config.target_simulator}')

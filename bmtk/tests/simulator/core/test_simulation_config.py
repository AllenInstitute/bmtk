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
import pytest
import os
import tempfile

from bmtk.simulator.core.simulation_config import SimulationConfig


def test_build_env():
    tmp_dir = tempfile.mkdtemp()  # tempfile.TemporaryDirectory()
    output_dir = os.path.join(tmp_dir, 'output')
    sim_config = {
        'manifest': {
            '$HOME_DIR': tmp_dir  # .name
        },
        'output': {
            'output_dir': '${HOME_DIR}/output',
            'log_file': 'simulation.log',
            'log_to_console': False
        }
    }

    cfg = SimulationConfig.from_dict(sim_config)
    cfg.build_env()
    assert(isinstance(cfg, SimulationConfig))
    assert(os.path.exists(output_dir))
    assert(os.path.exists(os.path.join(output_dir, 'simulation.log')))
    assert(os.path.exists(os.path.join(output_dir, 'sonata_config.json')))


def test_overwrite():
    tmp_dir = tempfile.mkdtemp()  # tempfile.TemporaryDirectory()
    output_dir = os.path.join(tmp_dir, 'output')
    sim_config = {
        'manifest': {
            '$HOME_DIR': tmp_dir
        },
        'output': {
            'output_dir': '${HOME_DIR}/output',
            'log_file': 'simulation.log',
            'log_to_console': False,
            'overwrite_output_dir': False
        }
    }

    cfg = SimulationConfig.from_dict(sim_config)
    cfg.build_env()
    assert(isinstance(cfg, SimulationConfig))
    assert(os.path.exists(output_dir))
    assert(os.path.exists(os.path.join(output_dir, 'simulation.log')))
    assert(os.path.exists(os.path.join(output_dir, 'sonata_config.json')))

    ## Changed behavior so that even if overwrite_output_dir is false and dir exists, it will still run
    # with pytest.raises(Exception):
    #     cfg = SimulationConfig.from_dict(sim_config)
    #     cfg.build_env()

    sim_config['output']['overwrite_output_dir'] = True
    cfg = SimulationConfig.from_dict(sim_config)
    cfg.build_env()
    assert(os.path.exists(output_dir))
    assert(os.path.exists(os.path.join(output_dir, 'simulation.log')))
    assert(os.path.exists(os.path.join(output_dir, 'sonata_config.json')))


if __name__ == '__main__':
    test_build_env()
    test_overwrite()

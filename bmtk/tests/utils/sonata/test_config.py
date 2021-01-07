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
import json
from datetime import datetime

from bmtk.utils.sonata.config import SonataConfig, from_json, from_dict


def test_json():
    config_file = tempfile.NamedTemporaryFile(suffix='.json')
    sonata_cfg = {
        'manifest': {'$BASE': '${configdir}', '$TMP_ATTR': 'mytest'},
        'myvar': '$TMP_ATTR/myvar',
        'cwd': '${workingdir}',
        'cdir': '${configdir}',
        'cfname': '${configfname}'
    }
    json.dump(sonata_cfg, open(config_file.name, 'w'))

    config_dict = SonataConfig.from_json(config_file.name)
    assert(isinstance(config_dict, SonataConfig))
    assert(isinstance(config_dict, dict))
    assert(config_dict['myvar'] == 'mytest/myvar')
    assert(config_dict['cwd'] == os.getcwd())
    assert(config_dict['cdir'] == os.path.dirname(config_file.name))
    assert(config_dict['cfname'] == config_file.name)

    config_dict = SonataConfig.load(config_file.name)
    assert(isinstance(config_dict, SonataConfig))
    assert(isinstance(config_dict, dict))
    assert(config_dict['myvar'] == 'mytest/myvar')
    assert(config_dict['cwd'] == os.getcwd())
    assert(config_dict['cdir'] == os.path.dirname(config_file.name))
    assert(config_dict['cfname'] == config_file.name)

    with pytest.warns(DeprecationWarning):
        config_dict = from_json(config_file.name)
        assert(isinstance(config_dict, SonataConfig))
        assert(isinstance(config_dict, dict))
        assert(config_dict['myvar'] == 'mytest/myvar')
        assert(config_dict['cwd'] == os.getcwd())
        assert(config_dict['cdir'] == os.path.dirname(config_file.name))
        assert(config_dict['cfname'] == config_file.name)


def test_json_split():
    """Can independently load a circuit_config.json and simulation_config.json into a single dict"""
    circuit_cfg = {
        'manifest': {'$NETWORK_DIR': 'network_tst'},
        'networks': {
            'node_files': {
                'nodes': '$NETWORK_DIR/nodes.h5',
                'node_types': '${NETWORK_DIR}/node_types.csv'
            }
        }
    }

    simulation_cfg = {
        'manifest': {'$OUTPUT_DIR': 'output_tst'},
        'output': {
            'output_dir': '$OUTPUT_DIR',
            'spikes_file': 'spikes.h5'
        }
    }

    circuit_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump(circuit_cfg, open(circuit_file.name, 'w'))

    # Case: circuit_cfg and simulation_cfg have been merged into a single json
    sim_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump(simulation_cfg, open(sim_file.name, 'w'))
    config_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump({
        'network':  circuit_file.name,
        'simulation': sim_file.name
    }, open(config_file.name, 'w'))
    config_dict = SonataConfig.from_json(config_file.name)
    assert(isinstance(config_dict, SonataConfig))
    assert(isinstance(config_dict, dict))
    assert(config_dict['output']['output_dir'] == 'output_tst')
    assert(config_dict['output']['spikes_file'] == 'output_tst/spikes.h5')
    assert(config_dict['networks']['node_files']['nodes'] == 'network_tst/nodes.h5')
    assert(config_dict['networks']['node_files']['node_types'] == 'network_tst/node_types.csv')

    # Case: one of the config files is missing
    sim_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump(simulation_cfg, open(sim_file.name, 'w'))
    config_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump({
        'simulation':  circuit_file.name
    }, open(config_file.name, 'w'))
    config_dict = SonataConfig.from_json(config_file.name)
    assert('output' not in config_dict)
    assert(config_dict['networks']['node_files']['nodes'] == 'network_tst/nodes.h5')
    assert(config_dict['networks']['node_files']['node_types'] == 'network_tst/node_types.csv')

    # Case: one config contains a link to another
    sim_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump(simulation_cfg, open(sim_file.name, 'w'))
    config_file = tempfile.NamedTemporaryFile(suffix='.json')
    simulation_cfg.update({'network': circuit_file.name})
    json.dump(simulation_cfg, open(config_file.name, 'w'))
    config_dict = SonataConfig.from_json(config_file.name)
    assert(config_dict['output']['output_dir'] == 'output_tst')
    assert(config_dict['output']['spikes_file'] == 'output_tst/spikes.h5')
    assert(config_dict['networks']['node_files']['nodes'] == 'network_tst/nodes.h5')
    assert(config_dict['networks']['node_files']['node_types'] == 'network_tst/node_types.csv')


def test_dict():
    sonata_dict = {
        'manifest': {'$BASE': '${configdir}', '$TMP_ATTR': 'mytest'},
        'myvar': '$TMP_ATTR/myvar'
    }

    config_dict = SonataConfig.from_dict(sonata_dict)
    assert(isinstance(config_dict, SonataConfig))
    assert(isinstance(config_dict, dict))
    assert(config_dict['myvar'] == 'mytest/myvar')

    config_dict = SonataConfig.load(sonata_dict)
    assert(isinstance(config_dict, SonataConfig))
    assert(isinstance(config_dict, dict))
    assert(config_dict['myvar'] == 'mytest/myvar')

    with pytest.warns(DeprecationWarning):
        config_dict = from_dict(sonata_dict)
        assert(isinstance(config_dict, SonataConfig))
        assert(isinstance(config_dict, dict))
        assert(config_dict['myvar'] == 'mytest/myvar')


def test_build_manifest1():
    """Test simple manifest"""
    config_file = {'manifest': {
        '$BASE_DIR': '/base',
        '$TMP_DIR': '$BASE_DIR/tmp',
        '$SHARE_DIR': '${TMP_DIR}_1/share'
    }}

    manifest = SonataConfig.from_dict(config_file)['manifest']
    assert(manifest['BASE_DIR'] == '/base')
    assert(manifest['TMP_DIR'] == '/base/tmp')
    assert(manifest['SHARE_DIR'] == '/base/tmp_1/share')


def test_build_manifest2():
    config_file = {'manifest': {
        '$DIR_DATA': 'data',
        '$DIR_MAT': 'mat',
        '$APPS': '/${DIR_DATA}/$DIR_MAT/apps'
    }}

    manifest = SonataConfig.from_dict(config_file)['manifest']
    assert(manifest['APPS'] == '/data/mat/apps')


def test_build_manifest_fail1():
    """Test exception occurs when variable is missing"""
    config_file = {'manifest': {
        '$BASE': '/base',
        '$TMP': '$VAR/Smat',
    }}
    with pytest.raises(Exception):
        SonataConfig.from_dict(config_file)


def test_build_manifest_fail2():
    """Test recursive definition"""
    config_file = {'manifest': {
        '$BASE': '$TMP/share',
        '$TMP': '$BASE/share',
    }}
    with pytest.raises(Exception):
        SonataConfig.from_dict(config_file)


def test_output_dir():
    cfg = SonataConfig.from_dict({
        'manifest': {'$OUTPUT_DIR': 'my/output'},
        'output': {
            'output_dir': '$OUTPUT_DIR',
            'log_file': 'log.txt',
            'spikes_file': 'tmp/spikes.h5',
            'spikes_file_csv': '/abs/path/to/spikes.csv',  # do not prepend to absolute paths
            'spikes_file_nwb': '$OUTPUT_DIR/spikes.nwb'  # do not prepend
        }
    })

    assert(cfg['output']['log_file'] == 'my/output/log.txt')
    assert(cfg['output']['spikes_file'] == 'my/output/tmp/spikes.h5')
    assert(cfg['output']['spikes_file_csv'] == '/abs/path/to/spikes.csv')
    assert(cfg['output']['spikes_file_nwb'] == 'my/output/spikes.nwb')


def test_speical_vars():
    cfg = SonataConfig.from_dict({
        'manifest': {
            '$VAR_DATETIME': '${datetime}'
        },
        'datetime': '${VAR_DATETIME}',
        'time': '${time}',
        'date': '${date}',
        'combined': 'myfile_${date}.csv'
    })

    assert(isinstance(datetime.strptime(cfg['datetime'], '%Y-%m-%d_%H-%M-%S'), datetime))
    assert(isinstance(datetime.strptime(cfg['time'], '%H-%M-%S'), datetime))
    assert(isinstance(datetime.strptime(cfg['date'], '%Y-%m-%d'), datetime))


def test_user_vars():
    cfg = SonataConfig.from_dict({
        'my_int': '${my_int}',
        'my_bool': '${my_bool}',
        'my_float': '${my_float}',
        'my_list': '${my_list}',
        'my_str': '${my_str}',
        'combined_strs': '${my_str}bar',
        'combined_int': 'file.${my_int}.txt'
    }, my_int=100, my_bool=True, my_float=0.001, my_list=['a', 'b'], my_str='foo')

    assert(cfg['my_int'] == 100)
    assert(cfg['my_bool'] is True)
    assert(cfg['my_float'] == 0.001)
    assert(cfg['my_list'] == ['a', 'b'])
    assert(cfg['my_str'] == 'foo')
    assert(cfg['combined_strs'] == 'foobar')
    assert(cfg['combined_int'] == 'file.100.txt')


def test_node_set_file():
    tmp_ns_file = tempfile.NamedTemporaryFile(suffix='.json')
    json.dump({
        'bio_cells': {'model': 'biophysical', 'locations': ['L4', 'L2/3']}
    }, open(tmp_ns_file.name, 'w'))

    cfg = SonataConfig.from_dict({
        'target_simulator': 'NEURON',
        'node_sets_file': tmp_ns_file.name
    })

    assert('node_sets' in cfg)
    assert('node_sets_file' in cfg)
    assert(set(cfg['node_sets'].keys()) == {'bio_cells'})
    assert(set(cfg['node_sets']['bio_cells'].keys()) == {'model', 'locations'})
    assert(cfg['node_sets']['bio_cells']['model'] == 'biophysical')
    assert(cfg['node_sets']['bio_cells']['locations'] == ['L4', 'L2/3'])

    cfg = SonataConfig.from_dict({
        'target_simulator': 'NEURON',
        'node_sets_file': tmp_ns_file.name,
        'node_sets': {'point_cells': {'key': 'val'}}
    })
    assert('node_sets' in cfg)
    assert('node_sets_file' in cfg)
    assert(set(cfg['node_sets']['point_cells'].keys()) == {'key'})


if __name__ == '__main__':
    # test_json()
    # test_json_split()
    # test_dict()
    # test_build_manifest1()
    # test_build_manifest2()
    # test_build_manifest_fail1()
    # test_build_manifest_fail2()
    # test_output_dir()
    # test_speical_vars()
    # test_user_vars()
    test_node_set_file()

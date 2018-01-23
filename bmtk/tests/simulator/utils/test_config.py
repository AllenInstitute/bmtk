import os
import pytest

import bmtk.simulator.utils.config as cfg


def config_path(rel_path):
    c_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(c_path, rel_path)


def test_load_parent_config():
    """Test a parent config file can pull in children configs"""
    cfg_full_path = config_path('files/config.json')
    config = cfg.from_json(cfg_full_path)
    assert(config['config_path'] == cfg_full_path)
    assert('components' in config)
    assert('networks' in config)
    assert('run' in config)


def test_load_network_config():
    cfg_full_path = config_path('files/circuit_config.json')
    config = cfg.from_json(cfg_full_path)
    manifest = config['manifest']
    assert(config['config_path'] == cfg_full_path)
    assert(config['components']['morphologies'] == os.path.join(manifest['$COMPONENT_DIR'], 'morphologies'))
    assert(config['networks']['node_files'][0]['nodes'] == os.path.join(manifest['$NETWORK_DIR'], 'V1/v1_nodes.h5'))


def test_load_simulator_config():
    cfg_full_path = config_path('files/simulator_config.json')
    config = cfg.from_json(cfg_full_path)
    manifest = config['manifest']
    assert('run' in config)
    assert(config['output']['log'] == os.path.join(manifest['$OUTPUT_DIR'], 'log.txt'))


def test_build_manifest1():
    """Test simple manifest"""
    config_file = {'manifest': {
        '$BASE_DIR': '/base',
        '$TMP_DIR': '$BASE_DIR/tmp',
        '$SHARE_DIR': '${TMP_DIR}_1/share'
    }}

    manifest = cfg.__build_manifest(config_file)
    assert(manifest['$BASE_DIR'] == '/base')
    assert(manifest['$TMP_DIR'] == '/base/tmp')
    assert(manifest['$SHARE_DIR'] == '/base/tmp_1/share')


def test_build_manifest2():
    config_file = {'manifest': {
        '$DIR_DATA': 'data',
        '$DIR_MAT': 'mat',
        '$APPS': '/${DIR_DATA}/$DIR_MAT/apps'
    }}

    manifest = cfg.__build_manifest(config_file)
    assert(manifest['$APPS'] == '/data/mat/apps')


def test_build_manifest_fail1():
    """Test exception occurs when variable is missing"""
    config_file = {'manifest': {
        '$BASE': '/base',
        '$TMP': '$VAR/Smat',
    }}
    with pytest.raises(Exception):
        cfg.__build_manifest(config_file)


def test_build_manifest_fail2():
    """Test recursive definition"""
    config_file = {'manifest': {
        '$BASE': '$TMP/share',
        '$TMP': '$BASE/share',
    }}
    with pytest.raises(Exception):
        cfg.__build_manifest(config_file)


def test_resolve_var_str():
    """Check that a variable can be resolved in a string"""
    config_file = {
        'manifest': {
            '$BASE': 'path'
        },
        's1': '$BASE/test',
        'i1': 9
    }
    conf = cfg.from_dict(config_file)
    assert(conf['s1'] == 'path/test')
    assert(conf['i1'] == 9)


def test_resolve_var_list():
    """Check variables can be resolved in list"""
    config_file = {
        'manifest': {
            '$p1': 'a',
            '$p2': 'b'
        },
        'l1': ['$p1/test', '${p2}/test', 9]
    }
    conf = cfg.from_dict(config_file)
    assert(conf['l1'][0] == 'a/test')
    assert(conf['l1'][1] == 'b/test')
    assert(conf['l1'][2] == 9)


def test_resolve_var_dict():
    """Check variables can be resolved in dictionary"""
    config_file = {
        'manifest': {
            '$v1': 'a',
            '$v2': 'c'
        },
        'd1': {
            'k1': '$v1',
            'k2': 'B',
            'k3': ['${v2}'],
            'k4': 4
        }
    }
    conf = cfg.from_dict(config_file)
    assert(conf['d1']['k1'] == 'a')
    assert(conf['d1']['k2'] == 'B')
    assert(conf['d1']['k3'] == ['c'])
    assert(conf['d1']['k4'] == 4)


def test_time_vars():
    config_file = {
        'd1': {
            'k1': 'k1_${date}',
            'k2': 'k2/$time',
            'k3': ['${datetime}'],
            'k4': 4
        }
    }

    conf = cfg.from_dict(config_file)



#test_time_vars()

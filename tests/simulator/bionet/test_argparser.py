import sys
import pytest

from bmtk.simulator.bionet import ArgumentParser


@pytest.mark.parametrize('args,parsed_output', [
    ('run_bionet.py', 'config.default.json'),
    ('run_bionet.py config.new.json', 'config.new.json'),
    ('nrniv -python run_bionet.py', 'config.default.json'),
    ('nrniv -python run_bionet.py config.new.json', 'config.new.json'),
    ('nrniv.exe -python run_bionet.py', 'config.default.json'),
    ('nrniv.exe -python run_bionet.py config.new.json', 'config.new.json'),
    ('nrniv.exe -python run_bionet.py --invalid', 'config.default.json'),
    ('nrniv.exe -python run_bionet.py config.new.json --invalid', 'config.new.json'),
])
def test_parse_args(monkeypatch, args, parsed_output):
    monkeypatch.setattr(sys, 'argv', args.split())
    parser = ArgumentParser()
    parser.add_argument('--opt', type=float, nargs=1, default='1.0')
    parser.add_argument('config_path', type=str, nargs='?', default='config.default.json')

    args, _ = parser.parse_known_args()
    assert(args.opt == 1.0)
    assert(args.config_path == parsed_output)

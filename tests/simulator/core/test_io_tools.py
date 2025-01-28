import os
import pytest
import tempfile

from bmtk.simulator.core.io_tools import IOUtils


def test_setup():
    io = IOUtils()
    tmp_dir = tempfile.mkdtemp()
    io.setup_output_dir(tmp_dir, log_file=None)
    assert(os.path.exists(tmp_dir))


def test_logging(capsys):
    io = IOUtils()
    io.log_info('hello')
    captured = capsys.readouterr()
    assert('hello' in captured.out)

    io.log_warning('test')
    captured = capsys.readouterr()
    assert('test' in captured.out)

    with pytest.raises(Exception):
        io.log_exception('boo')


if __name__  == '__main__':
    test_logging()
    # test_setup()

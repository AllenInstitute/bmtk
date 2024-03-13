import pytest
import numpy as np
import pandas as pd
import tempfile
from bmtk.simulator.core.modules import iclamp


@pytest.mark.parametrize('mod_args', [
    ({
        'amp': 0.01,
        'delay': 1000,
        'duration': 500.0
     }),
    ({
        'amp': [0.1, -0.1, 1.0, -1.0],
        'delay': [500.0, 1000.0, 1500.0, 2000.0],
        'duration': [500.0, 500.0, 500.0, 500.0]
    })

])
def test_AmpsReader(mod_args):
    reader = iclamp.AmpsReader(**mod_args)
    assert(reader.amps and isinstance(reader.amps, (list, tuple, np.ndarray)))
    assert(reader.delays and isinstance(reader.delays, (list, tuple, np.ndarray)))
    assert(reader.durations and isinstance(reader.durations, (list, tuple, np.ndarray)))


@pytest.mark.parametrize('mod_args', [
    ({
        'amp': [-10.0, 10.0],
        'delay': [1000.0, 2000.0]
    }),
    ({
        'amp': 10.0,
        'delay': [1000.0, 2000.0],
        'duration': [500.0, 500.0]
    }),
    ({
        'amp': [-10.0, 10.0],
        'delay': [1000.0, 2000.0],
        'duration': [500.0, 500.0, 500.0]
    })

])
def test_AmpsReader_invalid(mod_args):
    with pytest.raises(Exception):
        reader = iclamp.AmpsReader(**mod_args)


# def create_csv(amps, ts, amps_col='amps', ts_col='timestamps', sep=' '):
#     tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
#     pd.DataFrame({
#         ts_col: ts,
#         amps_col: amps
#     }).to_csv(tmpfile.name, sep=sep)
#     return tmpfile


# @pytest.mark.parametrize('ts,amps,expected_dt,expected_amps', [
#     (
#         np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0]),
#         np.array([0.0, -0.2, 0.0, 0.2, 0.0]),
#         500.0,
#         [0.0, -0.2, 0.0, 0.2, 0.0]
#     ),
#     (
#         np.array([0.0, 500.0, 1000.0, 1500.0]),
#         np.array([0.0, -0.2, 0.0, 0.2]),
#         500.0,
#         [0.0, -0.2, 0.0, 0.2]
#     ),
#     (
#         np.array([500.0, 1000.0, 1500.0, 2000.0]),
#         np.array([-0.2, 0.0, 0.2, 0.0]),
#         500.0,
#         [0.0, -0.2, 0.0, 0.2, 0.0]
#     ),
#     (
#         np.array([700.0, 1200.0, 1700.0, 2200.0]),
#         np.array([-0.2, 0.0, 0.2, 0.0]),
#         100.0,
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2,
#          0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
#     ),
#     (
#         np.array([300.0, 800.0, 1300.0, 1800.0]),
#         np.array([-0.2, 0.0, 0.2, 0.0]),
#         100.0,
#         [0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0,
#          0.0, 0.0]
#     ),
#     (
#         np.array([600.0, 800.0, 1000.0, 1200.0]),
#         np.array([.1, .2, 0.3, 0.0]),
#         200.0,
#         [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0]
#     )
# ])
# def test_recalc_timestamps(ts, amps, expected_dt, expected_amps):
#     tmpfile = create_csv(ts=ts, amps=amps)
#     csv_reader = iclamp.CSVAmpReader(file=tmpfile.name)

#     print(csv_reader.amps)
#     # assert(expected_dt == csv_reader.dt)
#     # assert(np.allclose(csv_reader.amps, expected_amps))


# if __name__ == '__main__':
#     test_recalc_timestamps(
#         ts=np.array([600.0, 800.0, 1000.0, 1200.0]), 
#         amps=np.array([.1, .2, 0.3, 0.0]), expected_dt=200.0, 
#         expected_amps=[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0]
#     )
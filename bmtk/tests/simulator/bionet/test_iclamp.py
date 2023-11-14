import pytest

import os
import numpy as np
import tempfile
import pandas as pd
from neuron import h
import h5py

try:
    from conftest import MORPH_DIR, load_neuron_modules
except (ModuleNotFoundError, ImportError)  as mnfe:
    from .conftest import MORPH_DIR, load_neuron_modules
    
from bmtk.simulator.bionet.modules import iclamp

pc = h.ParallelContext()


class NRNPythonObj(object):
    def post_fadvance(self):
        pass


h.pysim = NRNPythonObj()

RORB_SWC_PATH = os.path.join(MORPH_DIR, 'rorb_480169178_morphology.swc')


class MockCell(object):
    def __init__(self, swc_path=RORB_SWC_PATH):
        self.hobj = h.Biophys1(swc_path)


class MockBioSimulator(object):
    def __init__(self):
        self.net = self
        self.cell = None
        self.vm = None

    def get_node_set(self, _):
        return self

    def gids(self):
        return [0]

    def get_cell_gid(self, _):
        self.cell = MockCell(RORB_SWC_PATH)
        self.vm = h.Vector()
        self.vm.record(self.cell.hobj.soma[0](0.5)._ref_v)
        return self.cell


def create_csv(amps, ts, amps_col='amps', ts_col='timestamps', sep=' '):
    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    pd.DataFrame({
        ts_col: ts,
        amps_col: amps
    }).to_csv(tmpfile.name, sep=sep)
    return tmpfile


def run_sim(tstop, v_init):
    h.v_init = v_init
    h.tstop = tstop
    h.run(h.tstop)


@pytest.mark.parametrize('input_type', [
    'invalid', 'current_clamp', 'csv', 'file', 'nwb', 'allen'
])
def test_bad_input(input_type):
    with pytest.raises(Exception):
        iclamp.IClampMod(input_type=input_type)


def test_sim_scalar():
    params = {
        'input_type': 'current_clamp',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'amp': 0.15,
        'delay': 10.0,
        'duration': 10.0
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)

    h.v_init = -60.0
    h.tstop = 40.0
    h.run(h.tstop)
    assert(np.max(list(sim.vm)) > -60)


def test_sim_list():
    params = {
        'input_type': 'current_clamp',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'amp': [0.1, 0.2, 0.3],
        'delay': [10.0, 20.0, 30.0],
        'duration': [5.0, 5.0, 5.0]
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)

    h.v_init = -60.0
    h.tstop = 40.0
    h.run(h.tstop)
    assert (np.max(list(sim.vm)) > -60)

    params = {
        'input_type': 'current_clamp',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'amp': [0.1, 0.2, 0.3],
        'delay': [10.0, 20.0, 30.0],
        'duration': 5.0
    }

    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)


def test_sim_csv():
    # Test it out with default column name and separator
    tmpfile = create_csv(
        ts=[0.0, 10.0, 20.0, 30.0, 40.0],
        amps=[0.0, 0.1, 0.0, 0.2, 0.0]
    )
    params = {
        'input_type': 'csv',
        'node_set': 'biophys_cells',
        'file': tmpfile.name
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)
    run_sim(tstop=45.0, v_init=-60.0)
    assert(np.max(list(sim.vm)) > -60)

    # Test it out with different column names and separators
    tmpfile = create_csv(
        ts=[0.0, 10.0, 20.0, 30.0, 40.0],
        amps=[0.0, 0.1, 0.0, 0.2, 0.0],
        ts_col='times',
        amps_col='amplitudes',
        sep=','
    )
    params = {
        'input_type': 'file',
        'node_set': 'biophys_cells',
        'file': tmpfile.name,
        'separator': ',',
        'timestamps_column': 'times',
        'amplitudes_column': 'amplitudes'
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)
    run_sim(tstop=45.0, v_init=-60.0)
    assert(np.max(list(sim.vm)) > -60)


def test_invalid_csv():
    # Timestamps are not evenly spaced
    tmpfile = create_csv(
        ts=[0.0, 10.0, 30.0, 40.0],
        amps=[0.0, 0.1, 0.2, 0.0]
    )
    params = {
        'input_type': 'csv',
        'node_set': 'biophys_cells',
        'file': tmpfile.name
    }

    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)

    # Only one timestamps
    tmpfile = create_csv(
        ts=[10.0],
        amps=[0.1]
    )
    params = {
        'input_type': 'csv',
        'node_set': 'biophys_cells',
        'file': tmpfile.name
    }
    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)

    # Valid csv but column names are not as expected
    tmpfile = create_csv(
        ts=[0.0, 10.0, 20.0, 30.0, 40.0],
        amps=[0.0, 0.1, 0.0, 0.2, 0.0],
        ts_col='times',
        amps_col='amplitudes',
        sep=','
    )
    params = {
        'input_type': 'csv',
        'node_set': 'biophys_cells',
        'file': tmpfile.name
    }
    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)


def test_invalid_onset():
    tmpfile = create_csv(
        ts=[-10.0, 10.0, 30.0, 50.0],
        amps=[0.0, 0.1, 0.2, 0.0]
    )
    params = {
        'input_type': 'csv',
        'node_set': 'biophys_cells',
        'file': tmpfile.name
    }

    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)


def test_iclamp_nwb():
    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    with h5py.File(tmpfile.name, 'w') as h5:
        ts_grp = h5.create_group('/epochs/Sweep_1/stimulus/timeseries')
        ts_grp.create_dataset('starting_time', data=[1000.0])
        ts_grp['starting_time'].attrs['rate'] = 100000.0
        ts_grp.create_dataset('data', data=np.arange(0.0, 1.5e-10, step=1.0e-12), dtype=float)

    params = {
        'input_type': 'nwb',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'file': tmpfile.name,
        'sweep_id': 1,
        'delay': 10.0
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)

    h.v_init = -60.0
    h.tstop = 40.0
    h.run(h.tstop)
    assert(np.max(list(sim.vm)) > -60)

    # Test sweep id can be 'Sweep_1', 1, or '1'
    params = {
        'input_type': 'nwb',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'file': tmpfile.name,
        'sweep_id': '1',
        'delay': 10.0
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)

    # Test sweep id can be 'Sweep_1', 1, or '1'
    params = {
        'input_type': 'nwb',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'file': tmpfile.name,
        'sweep_id': 'Sweep_1',
        'delay': 10.0
    }

    ic = iclamp.IClampMod(**params)
    sim = MockBioSimulator()
    ic.initialize(sim)

    # Test sweep id can be 'Sweep_1', 1, or '1'
    params = {
        'input_type': 'nwb',
        'module': 'IClamp',
        'node_set': 'biophys_cells',
        'file': tmpfile.name,
        'sweep_id': 'one',
        'delay': 10.0
    }

    with pytest.raises(Exception):
        ic = iclamp.IClampMod(**params)
        sim = MockBioSimulator()
        ic.initialize(sim)


@pytest.mark.parametrize('ts,amps,expected_dt,expected_amps', [
    (
        np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0]),
        np.array([0.0, -0.2, 0.0, 0.2, 0.0]),
        500.0,
        [0.0, -0.2, 0.0, 0.2, 0.0]
    ),
    (
        np.array([0.0, 500.0, 1000.0, 1500.0]),
        np.array([0.0, -0.2, 0.0, 0.2]),
        500.0,
        [0.0, -0.2, 0.0, 0.2]
    ),
    (
        np.array([500.0, 1000.0, 1500.0, 2000.0]),
        np.array([-0.2, 0.0, 0.2, 0.0]),
        500.0,
        [0.0, -0.2, 0.0, 0.2, 0.0]
    ),
    (
        np.array([700.0, 1200.0, 1700.0, 2200.0]),
        np.array([-0.2, 0.0, 0.2, 0.0]),
        100.0,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2,
         0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    ),
    (
        np.array([300.0, 800.0, 1300.0, 1800.0]),
        np.array([-0.2, 0.0, 0.2, 0.0]),
        100.0,
        [0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0,
         0.0, 0.0]
    ),
    (
        np.array([600.0, 800.0, 1000.0, 1200.0]),
        np.array([.1, .2, 0.3, 0.0]),
        200.0,
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0]
    )
])
def test_recalc_timestamps(ts, amps, expected_dt, expected_amps):
    tmpfile = create_csv(ts=ts, amps=amps)
    csv_reader = iclamp.CSVAmpReaderNRN(file=tmpfile.name)

    assert(expected_dt == csv_reader._idt)
    assert(np.allclose(csv_reader.amps, expected_amps))


if __name__ == '__main__':
    # test_iclamp_scalar()
    # test_iclamp_list()
    # test_sim_csv()
    test_invalid_csv()
    # test_iclamp_nwb()

    # test_recalc_timestamps(
    #     ts=np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0]),
    #     amps=np.array([0.0, -0.2, 0.0, 0.2, 0.0]),
    #     expected_dt=500.0,
    #     expected_amps=[0.0, -0.2, 0.0, 0.2, 0.0]
    # )
    # test_recalc_timestamps(
    #     ts=np.array([500.0, 1000.0, 1500.0, 2000.0]),
    #     amps=np.array([-0.2, 0.0, 0.2, 0.0]),
    #     expected_dt=500.0,
    #     expected_amps=[0.0, -0.2, 0.0, 0.2, 0.0]
    # )
    # test_recalc_timestamps(
    #     ts=np.array([700.0, 1200.0, 1700.0, 2200.0]),
    #     amps=np.array([-0.2, 0.0, 0.2, 0.0]),
    #     expected_dt=100.0,
    #     expected_amps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2,
    #                    0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    # )
    #
    # test_recalc_timestamps(
    #     ts=np.array([300.0, 800.0, 1300.0, 1800.0]),
    #     amps=np.array([-0.2, 0.0, 0.2, 0.0]),
    #     expected_dt=100.0,
    #     expected_amps=[0.0, 0.0, 0.0, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2,
    #                    0.0, 0.0, 0.0, 0.0, 0.0]
    # )

    # test_recalc_timestamps(
    #     ts=np.array([600.0, 800.0, 1000.0, 1200.0]),
    #     amps=np.array([.1, .2, 0.3, 0.0]),
    #     expected_dt=200.0,
    #     expected_amps=[0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0]
    # )

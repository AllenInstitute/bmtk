import pytest
import logging
import pandas as pd
import numpy as np
import tempfile
import random

from bmtk.simulator.core.modules.ecephys_module import ECEphysUnitsModule
from bmtk.simulator.core.modules.ecephys_module import TimeWindow, NWBFileWrapper
from bmtk.simulator.core.modules.ecephys_module import MappingStrategy, UnitIdMapStrategy, SamplingStrategy

try:
    import pynwb
    has_pynwb = True
except ImportError as ie:
    has_pynwb = False


class MockNodeSet(object):
    def __init__(self, node_ids=[0, 1, 2, 3, 4]):
        self.node_ids = node_ids


def node_ids_map(with_times=True):
    map_df = pd.DataFrame({
        'node_ids': [0, 1, 2, 3, 4],
        'unit_ids': [99854, 99855, 99856, 99857, 99858] 
    })
    if with_times:
        map_df['start_times'] = [0.0, 500.0, 10000.0, 0.0, 5000.0]
        map_df['stop_times'] = [1000.0, 10000.0, 12500.0, 1000.0, 6000.0]
    
    return map_df


def build_nwb_file(init_id=99854, save_to_file=False):
    import pynwb
    from datetime import datetime
    from dateutil.tz import tzlocal

    nwbfile = pynwb.NWBFile(identifier=str(random.randint(10000000, 100000000-1)), session_start_time=datetime.now(tzlocal()), session_description='Ecephys session')
    device = nwbfile.create_device(name="probeA")
    nwbfile.add_electrode_column(name="label", description="label of electrode")
    eg = nwbfile.create_electrode_group(name='probeA', description='ECE Group', device=device, location="brain area")
    nwbfile.add_electrode(group=eg, id=8001, label="electrode", location="LGd", x=0.0, y=0.0, z=0.0)
    nwbfile.add_electrode(group=eg, id=8002, label="electrode", location="VISp", x=0.0, y=100.0, z=0.0)
    nwbfile.add_electrode(group=eg, id=8003, label="electrode", location="VISl", x=0.0, y=200.0, z=0.0)

    nwbfile.add_unit_column(name="quality", description="sorting quality")
    nwbfile.add_unit_column(name="channel_id", description="channel_id")
    nwbfile.add_unit_column(name="firing_rate", description="channel_id")
    nwbfile.add_unit(id=init_id, spike_times=[0.001, 0.002, 0.003, 0.004], quality='good', channel_id=8001, firing_rate=10.0)
    nwbfile.add_unit(id=init_id+1, spike_times=[0.500, 0.600, 0.700, 0.800, 0.900, 1.000], quality='good', channel_id=8002, firing_rate=10.0)
    nwbfile.add_unit(id=init_id+2, spike_times=[10.6663, 20.234, 50.22231, 102.312, 1004.1], quality='good', channel_id=8002, firing_rate=20.0)
    nwbfile.add_unit(id=init_id+3, spike_times=[], quality='bad', channel_id=8002, firing_rate=0.0)
    nwbfile.add_unit(id=init_id+4, spike_times=[5.11, 5.99, 6.18], quality='good', channel_id=8003, firing_rate=20.0)
    
    dg_intervals = pynwb.epoch.TimeIntervals(name="drifting_gratings_presentations", description="Stimulus")
    dg_intervals.add_column(name='temporal_frequency', description='temporal freq')
    dg_intervals.add_column(name='orientation', description='temporal freq')
    dg_intervals.add_row(start_time=10.0, stop_time=12.0, temporal_frequency=2.0, orientation=90.0)
    dg_intervals.add_row(start_time=20.0, stop_time=22.0, temporal_frequency=4.0, orientation=90.0)
    dg_intervals.add_row(start_time=30.0, stop_time=32.0, temporal_frequency=8.0, orientation=90.0)
    nwbfile.add_time_intervals(dg_intervals)

    if save_to_file:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nwb').name
        with pynwb.NWBHDF5IO(tmp_file, "w") as io:
            io.write(nwbfile)
        return tmp_file
    else:
        return nwbfile


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_basic():
    mod = ECEphysUnitsModule(
        name='test', mapping='units_map', node_set=MockNodeSet(), input_file=build_nwb_file(save_to_file=True), 
        units=node_ids_map()
    )
    assert(isinstance(mod._mapping_strategy, UnitIdMapStrategy))

    mod = ECEphysUnitsModule(
        name='test', mapping='units_map', node_set=MockNodeSet(), 
        input_file=[build_nwb_file(save_to_file=True), build_nwb_file(save_to_file=True)], 
        units=node_ids_map(with_times=False), interval=[10000.0, 20000.0]
    )
    assert(isinstance(mod._mapping_strategy, UnitIdMapStrategy))

    mod = ECEphysUnitsModule(
        name='test', mapping='sample', node_set=MockNodeSet(), 
        input_file=build_nwb_file(save_to_file=True), 
        filter={'location': 'VISp'}, interval=[10000.0, 20000.0]
    )
    assert(isinstance(mod._mapping_strategy, SamplingStrategy))
    
    mod = ECEphysUnitsModule(
        name='test', mapping='sample', node_set=MockNodeSet(), 
        input_file=build_nwb_file(save_to_file=True), 
        filter={'location': 'VISp'}, 
        interval={'interval_name': 'drifting_gratings', 'temporal_frequency': 2.0}
    )
    assert(isinstance(mod._mapping_strategy, SamplingStrategy))

    mod = ECEphysUnitsModule(
        name='test', mapping='sample_with_replacement', node_set=MockNodeSet(), 
        input_file=build_nwb_file(save_to_file=True), 
        filter={'location': 'LGd'}, 
        interval={'interval_name': 'drifting_gratings', 'temporal_frequency': 2.0}
    )
    assert(isinstance(mod._mapping_strategy, SamplingStrategy))

    with pytest.raises(Exception):
        mod = ECEphysUnitsModule(
            name='test', mapping='not_a_valid_mapping', node_set=MockNodeSet(), 
            input_file=build_nwb_file(save_to_file=True), 
            filter={'location': 'LGd'}, 
            interval={'interval_name': 'drifting_gratings', 'temporal_frequency': 2.0}
        )
        # assert(isinstance(mod._mapping_strategy, SamplingStrategy))


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_interval_defaults():
    nwb_file1 = NWBFileWrapper(build_nwb_file(save_to_file=True, init_id=0))
    nwb_file2 = NWBFileWrapper(build_nwb_file(save_to_file=True, init_id=100))
    tw = TimeWindow(None, [nwb_file1])
    assert(tw[0, nwb_file1.uuid] is None)

    tw = TimeWindow([], [nwb_file1])
    assert(tw[0, nwb_file1.uuid] is None)

    tw = TimeWindow([0.0, 1500.0], [nwb_file1])
    assert(np.allclose(tw[0, nwb_file1.uuid], [0.0, 1.5]))

    tw = TimeWindow([[0.0, 1500.0]], [nwb_file1])
    assert(np.allclose(tw[0, nwb_file1.uuid], [0.0, 1.5]))

    tw = TimeWindow([[0.0, 1500.0], np.array([2000.0, 2345.6])], [nwb_file1, nwb_file2])
    assert(np.allclose(tw[0, nwb_file1.uuid], [0.0, 1.5]))
    assert(np.allclose(tw[100, nwb_file2.uuid], [2.0, 2.3456]))

    tw = TimeWindow({'interval_name': 'drifting_gratings', 'interval_index': 0}, [nwb_file1])
    assert(np.allclose(tw[0, nwb_file1.uuid], [10.0, 12.0]))
    
    tw = TimeWindow([
        {'interval_name': 'drifting_gratings', 'interval_index': 1},
        {'interval_name': 'drifting_gratings_presentations', 'interval_index': 'all'}
        ], [nwb_file1, nwb_file2]
    )
    assert(np.allclose(tw[0,  nwb_file1.uuid], [20.0, 22.0]))
    assert(np.allclose(tw[0,  nwb_file2.uuid], [10.0, 32.0]))

    nwb_file3 = NWBFileWrapper(build_nwb_file(save_to_file=True, init_id=0))
    tw = TimeWindow([
        {'interval_name': 'drifting_gratings', 'interval_index': 0},
        [0.0, 500.0],
        {'interval_name': 'drifting_gratings_presentations', 'interval_index': 'all'}
        ], [nwb_file1, nwb_file2, nwb_file3]
    )
    assert(np.allclose(tw[0, nwb_file1.uuid], [10.0, 12.0]))
    assert(np.allclose(tw[1, nwb_file2.uuid], [0.0, 0.5]))
    assert(np.allclose(tw[2, nwb_file3.uuid], [10.0, 32.0]))
    

@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_unit_intervals_lu():
    nwb_file = NWBFileWrapper(build_nwb_file(save_to_file=False))
    tw = TimeWindow(defaults=[123.0, 456.0], nwb_files=[nwb_file])
    tw.units_lu = node_ids_map(with_times=True)
    assert(np.allclose(tw[99854, nwb_file.uuid], [0.0, 1.0]))
    assert(np.allclose(tw[99855, nwb_file.uuid], [0.5, 10.0]))
    assert(np.allclose(tw[99999, nwb_file.uuid], [0.123, 0.456]))

    nwb_file = NWBFileWrapper(build_nwb_file(save_to_file=True))
    tw = TimeWindow([123.0, 456.0], nwb_files=[nwb_file])
    tw.units_lu = node_ids_map(with_times=False).set_index('unit_ids')
    assert(np.allclose(tw[1000, nwb_file.uuid], [0.123, 0.456]))


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_load_map_strategy():
    spikes = MappingStrategy(input_file=build_nwb_file(init_id=99854, save_to_file=True))
    assert(len(spikes.units_table) == 5)

    spikes = MappingStrategy(input_file=[
        build_nwb_file(init_id=9000, save_to_file=True),
        build_nwb_file(init_id=10000, save_to_file=True)
    ])
    assert(len(spikes.units_table) == 10)

    spikes = MappingStrategy(input_file=build_nwb_file(init_id=99854, save_to_file=False))
    assert(len(spikes.units_table) == 5)

    spikes = MappingStrategy(input_file=[
        build_nwb_file(init_id=9000, save_to_file=False), build_nwb_file(init_id=10000, save_to_file=False)
    ])
    assert(len(spikes.units_table) == 10)


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_filter_units():
    nwbfile = build_nwb_file(init_id=99854, save_to_file=False)
    
    filter = {'location': 'VISp'}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 3)

    filter = {'location': 'VISp', 'quality': 'good'}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 2)

    filter = {'location': ['VISp', 'VISl']}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 4)

    filter = {'firing_rate': 20.0}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 2)

    filter = {'y': {'operation': '>=', 'value': 100.0}}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 4)

    filter = {
        'y_gt': {'column': 'y', 'operation': '>', 'value': 0.0},
        'y_lt': {'column': 'y', 'operation': '<', 'value': 200.0}
    }
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 3)

    filter = {'y': -100.0}
    spikes = MappingStrategy(input_file=nwbfile, units=filter)
    assert(len(spikes.units_table) == 0)

    with pytest.raises(Exception):
        filter = {'bad_col': 'x'}
        spikes = MappingStrategy(input_file=nwbfile, units=filter)
        spikes.units_table


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_unit_map_strategy():
    nwbfile = build_nwb_file(save_to_file=False)

    spikes = UnitIdMapStrategy(
        input_file=nwbfile, 
        units=node_ids_map(with_times=True)
    )
    
    spikes.build_map(node_set=None)
    assert(np.allclose(spikes.get_spike_trains(1, 'test_pop'), [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]))
    assert(np.allclose(spikes.get_spike_trains(3, 'test_pop'), []))
    assert(np.allclose(spikes.get_spike_trains(4, 'test_pop'), [110.0, 990.0]))

    spikes = UnitIdMapStrategy(
        interval=[500.0, 1000.0],
        input_file=nwbfile, 
        units=node_ids_map(with_times=False)
    )
    spikes.build_map(node_set=None)
    assert(np.allclose(spikes.get_spike_trains(0, 'test_pop'), []))
    assert(np.allclose(spikes.get_spike_trains(1, 'test_pop'), [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]))

    spikes = UnitIdMapStrategy(
        missing_ids='warn',
        input_file=nwbfile, 
        units=node_ids_map(with_times=False)
    )
    spikes.build_map(node_set=None)
    assert(np.allclose(spikes.get_spike_trains(100, 'test_pop'), []))

    with pytest.raises(Exception):
        spikes = UnitIdMapStrategy(
            missing_ids='fail',
            input_file=nwbfile, 
            units=node_ids_map(with_times=False)
        )
        spikes.build_map(node_set=None)
        spikes.get_spike_trains(100, 'test_pop')


@pytest.mark.skipif(not has_pynwb, reason='pynwb is not installed')
def test_sampling_strategy():
    nwbfile = build_nwb_file(save_to_file=False)

    spikes = SamplingStrategy(
        with_replacement=False,
        input_file=nwbfile
    )
    spikes.build_map(node_set=MockNodeSet([0, 1, 2, 3, 4]))
    assert(spikes.units2nodes_map)

    spikes = SamplingStrategy(
        with_replacement=True,
        input_file=nwbfile
    )
    spikes.build_map(node_set=MockNodeSet([0, 1, 2, 3, 4]))
    assert(spikes.units2nodes_map)

    spikes = SamplingStrategy(
        with_replacement=True,
        input_file=[build_nwb_file(init_id=0, save_to_file=False)]
    )
    spikes.build_map(node_set=MockNodeSet([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))
    assert(spikes.units2nodes_map)

    spikes = SamplingStrategy(
        with_replacement=True,
        input_file=[build_nwb_file(init_id=100, save_to_file=False), build_nwb_file(init_id=200, save_to_file=False)]
    )
    spikes.build_map(node_set=MockNodeSet([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))
    assert(spikes.units2nodes_map)

    with pytest.raises(Exception):
        spikes = SamplingStrategy(
            with_replacement=False,
            input_file=build_nwb_file(init_id=100, save_to_file=False)
        )
        spikes.build_map(node_set=MockNodeSet([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]))


if __name__ == '__main__':
    test_basic()
    test_interval_defaults()
    test_unit_intervals_lu()
    build_nwb_file()
    test_load_map_strategy()
    test_filter_units()
    test_unit_map_strategy()
    test_sampling_strategy()

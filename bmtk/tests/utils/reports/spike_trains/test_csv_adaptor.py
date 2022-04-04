import pytest
import numpy as np
import tempfile
import pandas as pd
from six import string_types

from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer, STCSVBuffer
from bmtk.utils.reports.spike_trains.spike_train_readers import CSVSTReader
from bmtk.utils.reports.spike_trains.spikes_file_writers import write_csv, write_csv_itr
from bmtk.utils.reports.spike_trains import sort_order


def create_st_buffer(st_cls):
    # Helper for creating spike_trains object
    if issubclass(st_cls, STCSVBuffer):
        return st_cls(cache_dir=tempfile.mkdtemp())
    else:
        return st_cls()


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv(st_cls, write_fnc):
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=0, timestamps=np.linspace(0, 1.0, 100))
    st.add_spikes(population='V1', node_ids=2, timestamps=np.linspace(2.0, 1.0, 10))
    st.add_spike(population='V1', node_id=1, timestamp=3.0)
    st.add_spikes(population='V2', node_ids=[3, 3, 3], timestamps=[0.25, 0.5, 0.75])

    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    write_fnc(tmpfile.name, st)

    df = pd.read_csv(tmpfile.name, sep=' ')
    assert(df.shape == (114, 3))
    assert(set(df.columns) == {'timestamps', 'population', 'node_ids'})
    assert(set(df['population'].unique()) == {'V1', 'V2'})

    assert(np.allclose(np.sort(df[(df['population'] == 'V1') & (df['node_ids'] == 0)]['timestamps']),
                       np.linspace(0, 1.0, 100), atol=1.0e-5))

    assert(np.allclose(np.sort(df[(df['population'] == 'V2') & (df['node_ids'] == 3)]['timestamps']),
                       [0.25, 0.5, 0.75]))


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv_bytime(st_cls, write_fnc):
    # Check we can sort by timestamps
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=0, timestamps=[0.5, 0.3, 0.1, 0.2, 0.4])

    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    write_fnc(tmpfile.name, st, sort_order=sort_order.by_time)
    df = pd.read_csv(tmpfile.name, sep=' ')
    assert(df.shape == (5, 3))
    assert(np.all(df['population'].unique() == 'V1'))
    assert(np.all(df['node_ids'].unique() == 0))
    assert(np.all(df['timestamps'] == [0.1, 0.2, 0.3, 0.4, 0.5]))


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv_byid(st_cls, write_fnc):
    # Check we can sort by node_ids
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=[2, 4, 2, 1, 3, 3, 6, 0], timestamps=[0.1]*8)

    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    write_fnc(tmpfile.name, st, sort_order=sort_order.by_id)
    df = pd.read_csv(tmpfile.name, sep=' ')
    assert(df.shape == (8, 3))
    assert(np.all(df['population'].unique() == 'V1'))
    assert(np.all(df['node_ids'] == [0, 1, 2, 2, 3, 3, 4, 6]))
    assert(np.all(df['timestamps'] == [0.1]*8))


def test_csv_reader():
    df = pd.DataFrame({
        'node_ids': [0, 0, 0, 0, 2, 1, 2] + [10, 10, 10],
        'population': ['V1']*7 + ['V2']*3,
        'timestamps': [0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0] + [4.0, 4.0, 4.0]
    })

    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    df.to_csv(tmpfile.name, sep=' ', columns=['timestamps', 'population', 'node_ids'])

    st = CSVSTReader(path=tmpfile.name, default_population='V1')
    assert(set(st.populations) == {'V1', 'V2'})

    assert(st.n_spikes() == 7)
    assert(st.n_spikes(population='V1') == 7)
    assert(st.n_spikes(population='V2') == 3)

    assert(set(st.node_ids()) == {0, 1, 2})
    assert(set(st.node_ids(population='V1')) == {0, 1, 2})
    assert(np.all(st.node_ids(population='V2') == [10]))

    assert(np.allclose(np.sort(st.get_times(0)), [0.25, 0.50, 0.75, 1.0]))
    assert(np.allclose(st.get_times(1, population='V1'), [0.001]))
    assert(np.allclose(st.get_times(10, population='V2'), [4.0, 4.0, 4.0]))

    df = st.to_dataframe()
    assert(len(df) == 10)
    assert(set(df.columns) == {'timestamps', 'population', 'node_ids'})
    df = st.to_dataframe(populations='V1', sort_order=sort_order.by_id, with_population_col=False)
    assert(len(df) == 7)
    assert(set(df.columns) == {'timestamps', 'node_ids'})
    assert(np.all(np.diff(df['node_ids']) >= 0))

    all_spikes = list(st.spikes())
    assert(len(all_spikes) == 10)
    assert(isinstance(all_spikes[0][0], (float, float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], (int, np.uint, np.integer)))


def test_csv_reader_nopop():
    df = pd.DataFrame({
        'node_ids': [0, 0, 0, 0, 2, 1, 2] + [10, 10, 10],
        # 'population': ['V1']*7 + ['V2']*3,
        'timestamps': [0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0] + [4.0, 4.0, 4.0]
    })

    tmpfile = tempfile.NamedTemporaryFile(suffix='.csv')
    df.to_csv(tmpfile.name, sep=' ', header=False, index=False, columns=['timestamps', 'node_ids'])

    st = CSVSTReader(path=tmpfile.name, default_population='V1')
    assert(set(st.populations) == {'V1'})
    assert(st.n_spikes() == 10)
    assert(set(st.node_ids()) == {0, 1, 2, 10})

    assert(np.allclose(np.sort(st.get_times(0)), [0.25, 0.50, 0.75, 1.0]))
    assert(np.allclose(st.get_times(1, population='V1'), [0.001]))
    assert(np.allclose(st.get_times(10, population='V1'), [4.0, 4.0, 4.0]))

    df = st.to_dataframe()
    assert(len(df) == 10)
    assert(set(df.columns) == {'timestamps', 'population', 'node_ids'})
    df = st.to_dataframe(populations='V1', sort_order=sort_order.by_id, with_population_col=False)
    assert(len(df) == 10)
    assert(set(df.columns) == {'timestamps', 'node_ids'})
    assert(np.all(np.diff(df['node_ids']) >= 0))

    all_spikes = list(st.spikes())
    assert(len(all_spikes) == 10)
    assert(isinstance(all_spikes[0][0], (float, float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], (int, np.uint, np.integer)))


if __name__ == '__main__':
    # test_write_csv(STMemoryBuffer, write_csv)
    # test_write_csv(STMemoryBuffer, write_csv_itr)
    # test_write_csv_bytime(STMemoryBuffer, write_csv_itr)
    # test_write_csv_byid(STMemoryBuffer, write_csv)
    # test_csv_reader()
    test_csv_reader_nopop()

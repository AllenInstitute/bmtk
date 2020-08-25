import pytest
import numpy as np
import tempfile
import h5py
from six import string_types
import warnings

from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer, STCSVBuffer
from bmtk.utils.reports.spike_trains import sort_order, pop_na
from bmtk.utils.reports.spike_trains.spike_train_readers import load_sonata_file, SonataSTReader, SonataOldReader, EmptySonataReader
from bmtk.utils.reports.spike_trains.spikes_file_writers import write_sonata, write_sonata_itr
from bmtk.utils.sonata.utils import check_magic, get_version, add_hdf5_magic, add_hdf5_version


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
    write_sonata,
    write_sonata_itr
])
def test_write_sonata(st_cls, write_fnc):
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=0, timestamps=np.linspace(0, 1.0, 100))
    st.add_spikes(population='V1', node_ids=2, timestamps=np.linspace(2.0, 1.0, 10))
    st.add_spike(population='V1', node_id=1, timestamp=3.0)
    st.add_spikes(population='V2', node_ids=[3, 3, 3], timestamps=[0.25, 0.5, 0.75])

    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    write_fnc(tmpfile.name, st)

    with h5py.File(tmpfile.name, 'r') as h5:
        assert(check_magic(h5))
        assert(get_version(h5) is not None)
        assert('/spikes/V1' in h5)
        node_ids = h5['/spikes/V1/node_ids'][()]
        assert(len(node_ids) == 111)
        assert(set(np.unique(node_ids)) == {0, 1, 2})
        assert(len(h5['/spikes/V1/timestamps'][()]) == 111)

        assert('/spikes/V2' in h5)
        assert(np.all(h5['/spikes/V2/node_ids'][()] == [3, 3, 3]))
        # WARNING: Not all adaptor guarentee order of spikes
        assert(np.allclose(np.sort(h5['/spikes/V2/timestamps'][()]), [0.25, 0.50, 0.75]))


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_empty(st_cls, write_fnc):
    # Important use case, a valid simulation may run for a long time but not produce any spikes, make sure it doesn't
    # fail trying to write any empty set of spike-trains to h5
    st = create_st_buffer(st_cls)
    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    write_fnc(tmpfile.name, st)

    with h5py.File(tmpfile.name, 'r') as h5:
        assert(check_magic(h5))
        assert(get_version(h5) is not None)
        assert('/spikes' in h5)
        assert(len(h5['/spikes']) == 0)


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_append(st_cls, write_fnc):
    # Check that we can append spikes data to an existing sonata file. Currently it only works if /spikes/<pop_name>
    # does not already exists, since append to an h5 can be prohibitive. iI the future may want ot implement.
    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmpfile.name, 'w') as h5:
        h5.create_group('/spikes/V1')

    st = create_st_buffer(st_cls)
    st.add_spikes(population='V2', node_ids=0, timestamps=np.linspace(0, 1.0, 100))

    write_fnc(tmpfile.name, st, mode='a', sort_order=sort_order.by_id)
    with h5py.File(tmpfile.name, 'r') as h5:
        assert(check_magic(h5))
        assert(get_version(h5) is not None)
        assert('/spikes/V1' in h5)
        assert('/spikes/V2' in h5)
        assert(len(h5['/spikes/V2/node_ids']) == 100)
        assert(len(h5['/spikes/V2/timestamps']) == 100)

    # Throw error if same pop_name alrady exists
    tmpfile2 = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmpfile2.name, 'w') as h5:
        h5.create_group('/spikes/V2')

    with pytest.raises(ValueError):
        write_fnc(tmpfile2.name, st, mode='a', sort_order=sort_order.by_id)


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_bytime(st_cls, write_fnc):
    # Check we can sort by timestamps
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=0, timestamps=[0.5, 0.3, 0.1, 0.2, 0.4])

    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    write_fnc(tmpfile.name, st, sort_order=sort_order.by_time)
    with h5py.File(tmpfile.name, 'r') as h5:
        assert(check_magic(h5))
        assert(get_version(h5) is not None)
        assert(h5['/spikes/V1'].attrs['sorting'] == 'by_time')
        assert(np.all(h5['/spikes/V1/node_ids'][()] == [0, 0, 0, 0, 0]))
        assert(np.all(h5['/spikes/V1/timestamps'][()] == [0.1, 0.2, 0.3, 0.4, 0.5]))


@pytest.mark.parametrize('st_cls', [
    STMemoryBuffer,
    STCSVBuffer
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_byid(st_cls, write_fnc):
    # Check we can sort by node_ids
    st = create_st_buffer(st_cls)
    st.add_spikes(population='V1', node_ids=[2, 4, 2, 1, 3, 3, 6, 0], timestamps=[0.1]*8)

    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    write_fnc(tmpfile.name, st, sort_order=sort_order.by_id)
    with h5py.File(tmpfile.name, 'r') as h5:
        assert(check_magic(h5))
        assert(get_version(h5) is not None)
        assert(h5['/spikes/V1'].attrs['sorting'] == 'by_id')
        assert(np.all(h5['/spikes/V1/node_ids'][()] == [0, 1, 2, 2, 3, 3, 4, 6]))
        assert(np.all(h5['/spikes/V1/timestamps'][()] == [0.1]*8))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sonata_reader():
    # Test ability to read an existing sonata file
    tmp_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmp_h5.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        h5.create_dataset('/spikes/V1/node_ids', data=[0, 0, 0, 0, 2, 1, 2], dtype=np.uint)
        h5.create_dataset('/spikes/V1/timestamps', data=[0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0], dtype=np.double)
        h5.create_dataset('/spikes/V2/node_ids', data=[10, 10, 10], dtype=np.uint)
        h5.create_dataset('/spikes/V2/timestamps', data=[4.0, 4.0, 4.0], dtype=np.double)
        h5.create_group('/spikes/V3')

    st = SonataSTReader(path=tmp_h5.name, default_population='V1')
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
    assert(isinstance(all_spikes[0][0], (np.double, np.float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], (np.int, np.uint)))


def test_oldsonata_reader():
    # A special reader for an older version of the spikes format
    tmp_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmp_h5.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        h5.create_dataset('/spikes/gids', data=[0, 0, 0, 0, 2, 1, 2], dtype=np.uint)
        h5.create_dataset('/spikes/timestamps', data=[0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0], dtype=np.double)

    st = SonataOldReader(path=tmp_h5.name)
    assert(np.all(st.populations == [pop_na]))
    assert(st.n_spikes() == 7)
    assert(set(st.node_ids()) == {0, 1, 2})
    assert(np.allclose(np.sort(st.get_times(0)), [0.25, 0.50, 0.75, 1.0]))

    df = st.to_dataframe()
    assert(df.shape == (7, 3))
    assert(set(df.columns) == {'timestamps', 'population', 'node_ids'})

    all_spikes = list(st.spikes())
    assert(len(all_spikes) == 7)
    assert(isinstance(all_spikes[0][0], (np.double, np.float)))
    assert(all_spikes[0][1] == pop_na)
    assert(isinstance(all_spikes[0][2], (np.int, np.uint)))


def test_load_sonata():
    warnings.simplefilter("ignore", UserWarning)

    # Sonata adaptor's factory method
    tmp_sonata = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmp_sonata.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        h5.create_dataset('/spikes/V1/node_ids', data=[0, 0, 0, 0, 2, 1, 2], dtype=np.uint)
        h5.create_dataset('/spikes/V1/timestamps', data=[0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0], dtype=np.double)
        h5.create_dataset('/spikes/V2/node_ids', data=[10, 10, 10], dtype=np.uint)
        h5.create_dataset('/spikes/V2/timestamps', data=[4.0, 4.0, 4.0], dtype=np.double)
        h5.create_group('/spikes/V3')

    tmp_sonata_old = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmp_sonata_old.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        h5.create_dataset('/spikes/gids', data=[0, 0, 0, 0, 2, 1, 2], dtype=np.uint)
        h5.create_dataset('/spikes/timestamps', data=[0.25, 0.5, 0.75, 1.0, 3.0, 0.001, 2.0], dtype=np.double)

    tmp_sonata_empty = tempfile.NamedTemporaryFile(suffix='.h5')
    with h5py.File(tmp_sonata_empty.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        h5.create_group('/spikes/')

    sr = load_sonata_file(tmp_sonata.name)
    assert(isinstance(sr, SonataSTReader))

    sr = load_sonata_file(tmp_sonata_old.name)
    assert(isinstance(sr, SonataOldReader))

    sr = load_sonata_file(tmp_sonata_empty.name)
    assert(isinstance(sr, EmptySonataReader))


if __name__ == '__main__':
    # test_write_sonata(STMemoryBuffer(), write_sonata)
    # test_write_sonata(STMemoryBuffer(), write_sonata_itr)

    # test_write_sonata(STCSVBuffer(cache_dir=tempfile.mkdtemp()), write_sonata_itr)
    # test_write_sonata_empty()
    # test_write_sonata_bytime()
    # test_write_sonata_byid()
    # test_write_sonata_append()

    # test_old_populations('spike_files/spikes.old.h5')
    # test_single_populations('spike_files/spikes.one_pop.h5')
    # test_multi_populations('spike_files/spikes.multipop.h5')
    # test_multipop_with_default('spike_files/spikes.multipop.h5')
    # test_empty_spikes()

    # test_sonata_reader()
    # test_oldsonata_reader()
    test_load_sonata()

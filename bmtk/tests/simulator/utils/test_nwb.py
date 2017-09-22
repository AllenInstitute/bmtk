import pytest
import numpy as np
from bmtk.simulator.utils import nwb
import os
import h5py


def test_create_blank_file():
    nwb.create_blank_file()
    f = nwb.create_blank_file(close=False)
    file_name = f.filename
    f.close()
    
    nwb.create_blank_file(file_name, force=True)
    os.remove(file_name)
    
    
def test_create_blank_file_force():
    temp_file_name = nwb.get_temp_file_name()
    nwb.create_blank_file(temp_file_name, force=True)
    try:
        nwb.create_blank_file(temp_file_name)
    except IOError:
        exception_caught = True
        assert exception_caught
    os.remove(temp_file_name)
    

def test_different_scales():
    y_values = 10*np.ones(10)
    f = nwb.create_blank_file(close=False)
    scale = nwb.DtScale(.1, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_stimulus(f)

    spike_train = nwb.FiringRate.get_stimulus(f, 0)
    y_values_new = spike_train.data[:]
    np.testing.assert_almost_equal(y_values_new, y_values)
    f.close()

    
def test_set_data_file_handle():
    f = nwb.create_blank_file(close=False)
    s0 = nwb._set_scale(f, '0', np.arange(3), 'time', 'second', "Scale")
    s1 = nwb._set_scale(f, '1', np.arange(4), 'time', 'second', "Scale")
    s2 = nwb._set_scale(f, '2', np.arange(5), 'time', 'second', "Scale")
    nwb._set_data(f, '1D', np.zeros(3), s0, 'firing_rate', 'hertz')
    nwb._set_data(f, '2D', np.zeros((3, 4)), (s0, s1), 'firing_rate', 'hertz')
    nwb._set_data(f, '3D', np.zeros((3, 4, 5)), (s0, s1, s2), 'firing_rate', 'hertz')
              
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_set_data_force():
    f = nwb.create_blank_file(close=False)
    s0 = nwb._set_scale(f, '0', np.arange(3), 'time', 'second', "Scale")
    nwb._set_data(f, 'test_force', np.zeros(3), s0, 'firing_rate', 'hertz')
    nwb._set_data(f, 'test_force', np.zeros(3), s0, 'firing_rate', 'hertz', force=True)

    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_get_data():
    s0_tuple = '0', np.arange(3), 'distance', 'pixel', "Scale"
    s1_tuple = '1', np.arange(4), 'distance', 'pixel', "Scale"
    data, dimension, unit = np.ones((3, 4)), 'brightness', 'intensity'
    
    f = nwb.create_blank_file(close=False)
    s0 = nwb._set_scale(f, *s0_tuple)
    s1 = nwb._set_scale(f, *s1_tuple)
    scales = (s0, s1)
    nwb._set_data(f, 'test', data, scales, dimension, unit)
    data_new, scales_new, dimension_new, unit_new, metadata = nwb._get_data(f['test'])
    np.testing.assert_almost_equal(data, data_new)
    assert len(metadata) == 0
    assert dimension == dimension_new
    assert unit == unit_new
    for scale_tuple, scale_new in zip((s0_tuple, s1_tuple), scales_new):
        np.testing.assert_almost_equal(scale_tuple[1], scale_new[:])
        assert scale_tuple[2] == scale_new.attrs['dimension']
        assert scale_tuple[3] == scale_new.attrs['unit']
        
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_metadata():
    f = nwb.create_blank_file(close=False)
    s0 = nwb._set_scale(f, '0', np.arange(3), 'time', 'second', "Scale")
    nwb._set_data(f, 'test_metadata', np.zeros(3), s0, 'firing_rate', 'hertz', metadata={'name':'foo'})
    _, _, _, _, metadata = nwb._get_data(f['test_metadata'])
    assert metadata['name'] == 'foo'
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_add_shared_scale():
    f = nwb.create_blank_file(close=False, force=True)
    t_values = np.arange(10)
    shared_scale = nwb.Scale(t_values, 'time', 'second')
    data_0 = nwb.FiringRate(10*np.ones(10), scale=shared_scale)
    data_0.add_to_stimulus(f)
    data_1 = nwb.FiringRate(20*np.ones(10), scale=shared_scale)
    data_1.add_to_stimulus(f)
    
    round_trip_0 = nwb.FiringRate.get_stimulus(f, 0)
    assert data_0 == round_trip_0
    
    round_trip_1 = nwb.FiringRate.get_stimulus(f, 1)
    assert data_1 == round_trip_1
    
    rt0, rt1 = nwb.FiringRate.get_stimulus(f)
    assert data_0 == rt0
    assert data_1 == rt1
    
    file_name = f.filename
    f.close()
    os.remove(file_name)
    

def test_firing_rate():
    t_values = np.arange(10)
    y_values = 10*np.ones(10)
    
    f = nwb.create_blank_file(close=False, force=True)

    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_stimulus(f)
    data.add_to_acquisition(f)
    data.add_to_processing(f, 'step_0')
    data.add_to_analysis(f, 'step_0')
    
    round_trip = nwb.FiringRate.get_stimulus(f, 0)
    assert round_trip == data
    
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_spike_train():
    t_values = np.arange(5)*.1
    y_values = np.array([0, 1, 2, 2, 1])
    
    f = nwb.create_blank_file(close=False, force=True)
    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.SpikeTrain(y_values, scale=scale)
    data.add_to_stimulus(f)
    data.add_to_acquisition(f)
    data.add_to_processing(f, 'step_0')
    data.add_to_analysis(f, 'step_0')
    
    round_trip = nwb.SpikeTrain.get_stimulus(f, 0)
    assert round_trip == data
    
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_grayscale_movie():
    t_values = np.arange(20)*.1
    row_values = np.arange(5)
    col_values = np.arange(10)
    data_values = np.empty((20, 5, 10))

    f = nwb.create_blank_file(close=False, force=True)
    t_scale = nwb.Scale(t_values, 'time', 'second')
    row_scale = nwb.Scale(row_values, 'distance', 'pixel')
    col_scale = nwb.Scale(col_values, 'distance', 'pixel')
    
    data = nwb.GrayScaleMovie(data_values, scale=(t_scale, row_scale, col_scale), metadata={'foo': 5})
    data.add_to_stimulus(f)
    data.add_to_acquisition(f)
    data.add_to_processing(f, 'step_0')
    data.add_to_analysis(f, 'step_0')
        
    round_trip = nwb.GrayScaleMovie.get_stimulus(f, 0)
    np.testing.assert_almost_equal(round_trip.data[:], data.data[:], 12)
    
    round_trip = nwb.GrayScaleMovie.get_acquisition(f, 0)
    np.testing.assert_almost_equal(round_trip.data[:], data.data[:], 12)
    
    round_trip = nwb.GrayScaleMovie.get_processing(f, 'step_0', 0)
    np.testing.assert_almost_equal(round_trip.data[:], data.data[:], 12)
    
    round_trip = nwb.GrayScaleMovie.get_analysis(f, 'step_0', 0)
    np.testing.assert_almost_equal(round_trip.data[:], data.data[:], 12)
    f.close()

    
def test_processing():
    t_values = np.arange(10)
    y_values = 10*np.ones(10)
    
    f = nwb.create_blank_file(close=False)

    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_processing(f, 'step_0')
    
    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_processing(f, 'step_0')
    
    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_processing(f, 'step_1')
        
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_analysis():
    t_values = np.arange(10)
    y_values = 10*np.ones(10)
    
    f = nwb.create_blank_file(close=False)

    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_analysis(f, 'step_0')
    
    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_analysis(f, 'step_0')
    
    scale = nwb.Scale(t_values, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    data.add_to_analysis(f, 'step_1')
        
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_writable():
    y_values = 10*np.ones(10)
    scale = nwb.DtScale(.1, 'time', 'second')
    data = nwb.FiringRate(y_values, scale=scale)
    
    f = nwb.create_blank_file(close=True)
    try:
        data.add_to_stimulus(f)
    except TypeError as e:
        assert str(e).replace('\'', '') == "NoneType object has no attribute __getitem__"
        
    f = nwb.create_blank_file(close=False)
    f.close()
    try:
        data.add_to_stimulus(f)
    except Exception as e:
        assert str(e) == 'File not valid: <Closed HDF5 file>'
        

def test_nullscale():
    y_values = np.array([.1, .5, .51])
    
    f = nwb.create_blank_file(force=True)
    data = nwb.SpikeTrain(y_values, unit='second')
    data.add_to_stimulus(f)

    spike_train = nwb.SpikeTrain.get_stimulus(f)
    y_values_new = spike_train.data[:]
    np.testing.assert_almost_equal(y_values, y_values_new)
    assert isinstance(spike_train.scales[0], nwb.NullScale)
    f.close()


def test_timeseries():
    y_values = np.array([.1, .2, .1])
    f = nwb.create_blank_file()
    scale = nwb.DtScale(.1, 'time', 'second')
    nwb.TimeSeries(y_values, scale=scale, dimension='voltage', unit='volt').add_to_acquisition(f)

    data = nwb.TimeSeries.get_acquisition(f)
    assert data.scales[0].dt == .1 
    assert data.scales[0].unit == 'second'
    np.testing.assert_almost_equal(data.data[:], y_values)
    assert data.unit == 'volt'
 
    file_name = f.filename
    f.close()
    os.remove(file_name)


def test_external_link():
    data_original = np.zeros(10)
    f = nwb.create_blank_file(force=True)
    scale = nwb.Scale(np.zeros(10), 'time', 'second')
    nwb.TimeSeries(data_original, scale=scale, dimension='voltage', unit='volt',
                   metadata={'foo': 1}).add_to_acquisition(f)
    temp_file_name = f.filename
    f.close()
    
    f = h5py.File(temp_file_name, 'r')
    f2 = nwb.create_blank_file(force=True)
    data = nwb.TimeSeries.get_acquisition(f, 0)
    data.add_to_acquisition(f2)
    f.close()
    temp_file_name_2 = f2.filename
    f2.close()
    
    f = h5py.File(temp_file_name_2)
    data = nwb.TimeSeries.get_acquisition(f, 0)
    np.testing.assert_almost_equal(data.data, data_original)
    assert data.data.file.filename == temp_file_name

    f.close()
    os.remove(temp_file_name)
    os.remove(temp_file_name_2)


if __name__ == "__main__":
    test_create_blank_file()        # pragma: no cover
    test_create_blank_file_force()  # pragma: no cover
    test_set_data_file_handle()     # pragma: no cover
    test_set_data_force()           # pragma: no cover
    test_get_data()                 # pragma: no cover
    test_metadata()                 # pragma: no cover
    test_add_shared_scale()         # pragma: no cover
    test_firing_rate()              # pragma: no cover
    test_processing()               # pragma: no cover
    test_analysis()                 # pragma: no cover
    test_spike_train()              # pragma: no cover
    test_grayscale_movie()          # pragma: no cover
#     test_get_stimulus()             # pragma: no cover
    test_different_scales()
    test_writable()
    test_nullscale()
    test_timeseries()
    test_external_link()

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
import copy
import numpy as np
import os
import h5py
import time
import uuid
import tempfile
from bmtk.analyzer.visualization.widgets import PlotWidget, MovieWidget

__version__ = '0.1.0'

allowed_dimensions = {'firing_rate': ('hertz',),
                      'time': ('second', 'millisecond'),
                      'brightness': ('intensity',),
                      'distance': ('pixel',),
                      'index': ('gid',),
                      'intensity': ('bit',None),
                      'voltage': ('volt',),
                      'current': ('ampere',),
                      None: (None,),
                      'dev': ('dev',)}

allowed_groups = {'firing_rate': ('firing_rate',),
                  'spike_train': ('index', 'time'),
                  'grayscale_movie': ('intensity',),
                  'time_series': ('voltage', 'current'),
                  'dev': ('dev',)}

top_level_data = ['file_create_date',
                  'stimulus',
                  'acquisition', 
                  'analysis', 
                  'processing',
                  'epochs', 
                  'general', 
                  'session_description',
                  'nwb_version',
                  'identifier']


def open_file(file_name):
    return h5py.File(file_name)


class Scale(object):
    def __init__(self, scale_range, dimension, unit):
        assert dimension in allowed_dimensions
        assert unit in allowed_dimensions[dimension]
        
        self.scale_range = scale_range
        self.dimension = dimension
        self.unit = unit
        self._hdf5_location = None
        
    def __eq__(self, other):
        d = self.dimension == other.dimension
        u = self.unit == other.unit
        s = np.allclose(self.scale_range, other.scale_range)
        return d and u and s
    
    @ property
    def data(self):
        return self.scale_range

    
class DtScale(object):
    def __init__(self, dt, dimension, unit):
        assert dimension in allowed_dimensions
        assert unit in allowed_dimensions[dimension]
        
        self.dt = dt
        self.dimension = dimension
        self.unit = unit
        self._hdf5_location = None
        
    def __eq__(self, other):
        d = self.dimension == other.dimension
        u = self.unit == other.unit
        s = np.allclose(self.scale_range, other.scale_range)
        return d and u and s
    
    @ property
    def data(self):
        return self.dt


class NullScale(object):
    
    def __init__(self):
        self._hdf5_location = None
        self.data = None 
        self.dimension = None
        self.unit = None


class Data(object):
    def __init__(self, data, dimension, unit, scales, metadata):
        assert dimension in allowed_dimensions
        assert unit in allowed_dimensions[dimension]
        if isinstance(scales, (Scale, DtScale)):
            assert len(data.shape) == 1
            scales = (scales,)
        
        for key in metadata.iterkeys():
            assert isinstance(key, (str, unicode))
        for ii, scale in enumerate(scales):
            if isinstance(scale, Scale):
                assert len(scale.scale_range) == data.shape[ii]
            elif isinstance(scale, DtScale):
                assert isinstance(scale.dt, (float, np.float)) and scale.dt > 0
            else:
                raise Exception
            
        if len(scales) == 0:
            scales = [NullScale()]
            
        metadata = copy.copy(metadata)
        self.data = data
        self.scales = scales
        self.dimension = dimension
        self.unit = unit
        self.metadata = metadata
        self._hdf5_location = None
        
    def __eq__(self, other):
        da = np.allclose(self.data, other.data)
        d = self.dimension == other.dimension
        u = self.unit == other.unit
        s = [s1 == s2 for s1, s2 in zip(self.scales, other.scales)].count(True) == len(self.scales)
        if len(self.metadata) != len(other.metadata):
            m = False
        else:
            try:
                sum = 0
                for key in self.metadata.keys():
                    sum += other.metadata[key] == self.metadata[key]
                assert sum == len(self.metadata)
                m = True
            except:
                m = False
        return da and d and u and s and m
    
    @staticmethod
    def _get_from_group(object_class, parent_group, group_name, ii=0):

        data_group = parent_group['%s/%s' % (group_name, ii)]
        data, scales, dimension, unit, metadata = _get_data(data_group)
        
        assert dimension in allowed_groups[object_class.group] 
         
        if unit == "None":
            unit = None
        scale_list = []
        for scale in scales:
            if scale.attrs['type'] == 'Scale':
                curr_scale = Scale(scale, scale.attrs['dimension'], scale.attrs['unit'])
            elif scale.attrs['type'] == 'DtScale':
                curr_scale = DtScale(float(scale.value), scale.attrs['dimension'], scale.attrs['unit'])
            elif scale.attrs['type'] == 'NullScale':
                curr_scale = None
            else:
                raise Exception
            if curr_scale is not None:
                scale_list.append(curr_scale)
        
        if len(scale_list) == 1:
            scale_list = scale_list[0]

        return object_class(data, dimension=dimension, unit=unit, scale=scale_list, metadata=metadata)

    def add_to_stimulus(self, f, compression='gzip', compression_opts=4):
        self._add_to_group(f, 'stimulus', self.__class__.group, compression=compression,
                           compression_opts=compression_opts)

    @classmethod
    def get_stimulus(cls, f, ii=None):
        if ii is None:
            return_data = [cls.get_stimulus(f, ii) for ii in range(len(f['stimulus/%s' % cls.group]))]
            if len(return_data) == 1:
                return_data = return_data[0]
            return return_data
        else:
            return Data._get_from_group(cls, f['stimulus'], cls.group, ii=ii)

    def add_to_acquisition(self, f, compression='gzip', compression_opts=4):
        self._add_to_group(f, 'acquisition', self.__class__.group, compression=compression,
                           compression_opts=compression_opts)

    @classmethod
    def get_acquisition(cls, f, ii=None):
        if ii is None:
            return_data = [cls.get_acquisition(f, ii) for ii in range(len(f['acquisition/%s' % cls.group]))]
            if len(return_data) == 1:
                return_data = return_data[0]
            return return_data

        else:
            return Data._get_from_group(cls, f['acquisition'], cls.group, ii=ii)
        
    def add_to_processing(self, f, processing_submodule_name):
        if processing_submodule_name not in f['processing']:
            f['processing'].create_group(processing_submodule_name)
        return self._add_to_group(f, 'processing/%s' % processing_submodule_name, self.__class__.group)
        
    @classmethod
    def get_processing(cls, f, subgroup_name, ii=None):
        if ii is None:
            return_data = {}
            for ii in range(len(f['processing/%s/%s' % (subgroup_name, cls.group)])):
                return_data[ii] = cls.get_processing(f, subgroup_name, ii)
            return return_data

        else:
            return Data._get_from_group(cls, f['processing/%s' % subgroup_name], cls.group, ii=ii)
        
    def add_to_analysis(self, f, analysis_submodule_name):
        if analysis_submodule_name not in f['analysis']:
            f['analysis'].create_group(analysis_submodule_name)
        return self._add_to_group(f, 'analysis/%s' % analysis_submodule_name, self.__class__.group)
        
    @classmethod
    def get_analysis(cls, f, subgroup_name, ii=None):
        if ii is None:
            return [cls.get_analysis(f, ii, subgroup_name)
                    for ii in range(len(f['analysis/%s/%s' % (subgroup_name, cls.group)]))]
        else:
            return Data._get_from_group(cls, f['analysis/%s' % subgroup_name], cls.group, ii=ii)

    def _add_to_group(self, f, parent_name, group_name, compression='gzip', compression_opts=4):
        assert group_name in allowed_groups
        assert self.dimension in allowed_groups[group_name]
        try:
            parent_group = f[parent_name]
        except ValueError:
            try:
                file_name = f.filename
                raise Exception('Parent group:%s not found in file %s' % parent_name, file_name)
            except ValueError:
                raise Exception('File not valid: %s' % f)
                
        if self.__class__.group in parent_group:
            subgroup = parent_group[self.__class__.group]
            int_group_name = str(len(subgroup))
        else:
            subgroup = parent_group.create_group(self.__class__.group)
            int_group_name = '0'
        
        # Create external link:
        if isinstance(self.data, h5py.Dataset):
            if subgroup.file == self.data.file:
                raise NotImplementedError
            else:
                return _set_data_external_link(subgroup, int_group_name, self.data.parent)
        else:
            dataset_group = subgroup.create_group(int_group_name)
        
        # All this to allow do shared scale management:
        scale_group = None
        scale_list = []
        for ii, scale in enumerate(self.scales):
            if isinstance(scale, (Scale, DtScale, NullScale)):
                if scale._hdf5_location is None:
                    if scale_group is None:
                        scale_group = dataset_group.create_group('scale')
                        curr_scale = _set_scale(scale_group, 'dimension_%s' % ii, scale.data, scale.dimension,
                                                scale.unit, scale.__class__.__name__)
                        scale._hdf5_location = curr_scale
                    else:
                        curr_scale = _set_scale(scale_group, 'dimension_%s' % ii, scale.data, scale.dimension,
                                                scale.unit, scale.__class__.__name__)
                        scale._hdf5_location = curr_scale
                else:
                    curr_scale = scale._hdf5_location
            elif isinstance(scale, h5py.Dataset):
                curr_scale = scale
            else:
                raise Exception

            scale_list.append(curr_scale)

        _set_data(subgroup, dataset_group.name, self.data, scale_list, self.dimension, self.unit,
                  metadata=self.metadata, compression=compression, compression_opts=compression_opts)


class FiringRate(Data):
    group = 'firing_rate'
    
    def __init__(self, data, **kwargs):
        dimension = 'firing_rate'
        unit = 'hertz'
        scale = kwargs.get('scale')
        metadata = kwargs.get('metadata', {})
        assert isinstance(scale, (Scale, DtScale))
        super(FiringRate, self).__init__(data, dimension, unit, scale, metadata)
        
    def get_widget(self, **kwargs):
        rate_data = self.data[:]
        t_range = self.scales[0].data[:]
        return PlotWidget(t_range, rate_data, metadata=self.metadata, **kwargs)


class Dev(Data):
    group = 'dev'

    def __init__(self, data, **kwargs):
        dimension = kwargs.get('dimension')
        unit = kwargs.get('unit')
        scale = kwargs.get('scale')
        metadata = kwargs.get('metadata', {})
        
        super(Dev, self).__init__(data, dimension, unit, scale, metadata)


class TimeSeries(Data):
    group = 'time_series'

    def __init__(self, data, **kwargs):
        dimension = kwargs.get('dimension')
        unit = kwargs.get('unit')
        scale = kwargs.get('scale')
        metadata = kwargs.get('metadata', {})
        
        assert isinstance(scale, (Scale, DtScale))
        assert scale.dimension == 'time'
        super(TimeSeries, self).__init__(data, dimension, unit, scale, metadata)
        
        
class SpikeTrain(Data):
    group = 'spike_train'

    def __init__(self, data, **kwargs):
        scales = kwargs.get('scale',[])
        unit = kwargs.get('unit', 'gid')
        metadata = kwargs.get('metadata',{})
        
        if isinstance(scales, Scale):
            super(SpikeTrain, self).__init__(data, 'index', unit, scales, metadata)
        elif len(scales) == 0:
            assert unit in allowed_dimensions['time']
            scales = []
            super(SpikeTrain, self).__init__(data, 'time', unit, scales, metadata)
        else:
            assert len(scales) == 1 and isinstance(scales[0], Scale)
            super(SpikeTrain, self).__init__(data, 'index', unit, scales, metadata)


class GrayScaleMovie(Data):
    group = 'grayscale_movie'
    
    def __init__(self, data, **kwargs):
        dimension = 'intensity'
        unit = kwargs.get('unit', None)
        scale = kwargs.get('scale')
        metadata = kwargs.get('metadata', {})
        
        super(GrayScaleMovie, self).__init__(data, dimension, unit, scale, metadata)
        
    def get_widget(self, ax=None):
        data = self.data[:]
        t_range = self.scales[0].data[:]
        return MovieWidget(t_range=t_range, data=data, ax=ax, metadata=self.metadata)


def get_temp_file_name():
    f = tempfile.NamedTemporaryFile(delete=False)
    temp_file_name = f.name
    f.close()
    os.remove(f.name)
    return temp_file_name


def create_blank_file(save_file_name=None, force=False, session_description='', close=False):

    if save_file_name is None:
        save_file_name = get_temp_file_name()

    if not force:
        f = h5py.File(save_file_name, 'w-')
    else:
        if os.path.exists(save_file_name):
            os.remove(save_file_name)
        f = h5py.File(save_file_name, 'w')       
    
    f.create_group('acquisition')
    f.create_group('analysis')
    f.create_group('epochs')
    f.create_group('general')
    f.create_group('processing')
    f.create_group('stimulus')
    
    f.create_dataset("file_create_date", data=np.string_(time.ctime()))
    f.create_dataset("session_description", data=session_description)
    f.create_dataset("nwb_version", data='iSee_%s' % __version__)
    f.create_dataset("identifier", data=str(uuid.uuid4()))

    if close:
        f.close()
    else:
        return f


def assert_subgroup_exists(child_name, parent):
    try:
        assert child_name in parent
    except:
        raise RuntimeError('Group: %s has no subgroup %s' % (parent.name, child_name))  
    

def _set_data_external_link(parent_group, dataset_name, data):
    parent_group[dataset_name] = h5py.ExternalLink(data.file.filename, data.name)
     

def _set_scale_external_link(parent_group, name, scale):
    print(parent_group, name, scale)
    print(scale.file.filename, scale.name)
    parent_group[name] = h5py.ExternalLink(scale.file.filename, scale.name)
    return parent_group[name]


def _set_data(parent_group, dataset_name, data, scales, dimension, unit, force=False, metadata={}, compression='gzip',
              compression_opts=4):
    # Check inputs:
    if isinstance(scales, h5py.Dataset):
        scales = (scales,)
    else:
        assert isinstance(scales, (list, tuple))

    assert data.ndim == len(scales)
    assert dimension in allowed_dimensions
    assert unit in allowed_dimensions[dimension]
    for ii, scale in enumerate(scales):
        assert len(scale.shape) in (0, 1)
        check_dimension = str(scale.attrs['dimension'])
        if check_dimension == 'None':
            check_dimension = None
        check_unit = scale.attrs['unit']
        if check_unit == 'None':
            check_unit = None
        assert check_dimension in allowed_dimensions
        assert check_unit in allowed_dimensions[check_dimension]
        if len(scale.shape) == 1:
            assert len(scale) == data.shape[ii] or len(scale) == 0

    if dataset_name not in parent_group:
        dataset_group = parent_group.create_group(dataset_name)
    else:
        dataset_group = parent_group[dataset_name]

    for key, val in metadata.iteritems():
        assert key not in dataset_group.attrs
        dataset_group.attrs[key] = val 

    if 'data' in dataset_group:
        if not force:
            raise IOError('Field "stimulus" of %s is not empty; override with force=True' % parent_group.name)
        else:
            del dataset_group['data']
    
    dataset = dataset_group.create_dataset(name='data', data=data, compression=compression,
                                           compression_opts=compression_opts)
    
    for ii, scale in enumerate(scales):
        dataset.dims[ii].label = scale.attrs['dimension']
        dataset.dims[ii].attach_scale(scale)
        
    dataset.attrs.create('dimension', str(dimension))
    dataset.attrs.create('unit', str(unit))
    
    return dataset


def _set_scale(parent_group, name, scale, dimension, unit, scale_class_name):
    assert dimension in allowed_dimensions
    assert unit in allowed_dimensions[dimension]
    
    if scale is None:
        scale = parent_group.create_dataset(name=name, shape=(0,))
    else:
        scale = np.array(scale)
        assert scale.ndim in (0, 1)
        scale = parent_group.create_dataset(name=name, data=scale)
    scale.attrs['dimension'] = str(dimension)
    scale.attrs['unit'] = str(unit)
    scale.attrs['type'] = scale_class_name

    return scale


def _get_data(dataset_group):
    data = dataset_group['data']
    dimension = dataset_group['data'].attrs['dimension']
    unit = dataset_group['data'].attrs['unit']
    scales = tuple([dim[0] for dim in dataset_group['data'].dims])
    metadata = dict(dataset_group.attrs)
    
    return data, scales, dimension, unit, metadata


def get_stimulus(f):
    category = 'stimulus'
    for parent_group in f[category]:
        for data_group in f[category][parent_group]:
            print(f[category][parent_group][data_group])


def add_external_links(parent_group, external_file_name, external_group_name_list=top_level_data):
    for subgroup in external_group_name_list:
        parent_group[subgroup] = h5py.ExternalLink(external_file_name, subgroup)

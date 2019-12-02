import os
import numpy as np
from .linearfilter import SpatioTemporalFilter
from .spatialfilter import GaussianSpatialFilter
from .temporalfilter import TemporalFilterCosineBump
from .movie import Movie
from .lgnmodel1 import LGNModel, heat_plot
from .transferfunction import MultiTransferFunction, ScalarTransferFunction
from .lnunit import LNUnit, MultiLNUnit
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y


class OnUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude > 0
        super(OnUnit, self).__init__(linear_filter, transfer_function)


class OffUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude < 0
        super(OffUnit, self).__init__(linear_filter, transfer_function)


class LGNOnOffCell(MultiLNUnit):
    """A cell model for a OnOff cell"""
    def __init__(self, on_filter, off_filter,
                 transfer_function=MultiTransferFunction((symbolic_x, symbolic_y), 'Heaviside(x)*(x)+Heaviside(y)*(y)')):
        """Summary

        :param on_filter:
        :param off_filter:
        :param transfer_function:
        """
        self.on_filter = on_filter
        self.off_filter = off_filter
        self.on_unit = OnUnit(self.on_filter, ScalarTransferFunction('s'))
        self.off_unit = OffUnit(self.off_filter, ScalarTransferFunction('s'))
        super(LGNOnOffCell, self).__init__([self.on_unit, self.off_unit], transfer_function)
        

class TwoSubfieldLinearCell(MultiLNUnit):
    def __init__(self, dominant_filter, nondominant_filter, subfield_separation=10, onoff_axis_angle=45,
                 dominant_subfield_location=(30,40),
                 transfer_function=MultiTransferFunction((symbolic_x, symbolic_y), 'Heaviside(x)*(x)+Heaviside(y)*(y)')):
        self.subfield_separation = subfield_separation
        self.onoff_axis_angle = onoff_axis_angle
        self.dominant_subfield_location = dominant_subfield_location
        self.dominant_filter = dominant_filter
        self.nondominant_filter = nondominant_filter
        self.transfer_function= transfer_function

        self.dominant_unit = LNUnit(self.dominant_filter, ScalarTransferFunction('s'), amplitude=self.dominant_filter.amplitude)
        self.nondominant_unit = LNUnit(self.nondominant_filter, ScalarTransferFunction('s'), amplitude=self.dominant_filter.amplitude)

        super(TwoSubfieldLinearCell, self).__init__([self.dominant_unit, self.nondominant_unit], self.transfer_function)
              
        self.dominant_filter.spatial_filter.translate = self.dominant_subfield_location
        hor_offset = np.cos(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation + self.dominant_subfield_location[0]
        vert_offset = np.sin(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation+ self.dominant_subfield_location[1]
        rel_translation = (hor_offset,vert_offset)
        self.nondominant_filter.spatial_filter.translate = rel_translation
        
        
class LGNOnCell(object):
    def __init__(self, **kwargs):
        self.position = kwargs.pop('position', None)
        self.weights = kwargs.pop('weights', None)
        self.kpeaks = kwargs.pop('kpeaks', None)
        self.amplitude = kwargs.pop('amplitude', None)
        self.sigma = kwargs.pop('sigma', None)
        self.transfer_function_str = kwargs.pop('transfer_function_str', 's')  # 'Heaviside(s)*s')
        self.metadata = kwargs.pop('metadata', {})

        temporal_filter = TemporalFilterCosineBump(self.weights, self.kpeaks)
        spatial_filter = GaussianSpatialFilter(translate=self.position, sigma=self.sigma,  origin=(0,0)) # all distances measured from BOTTOM LEFT
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=self.amplitude)
        transfer_function = ScalarTransferFunction(self.transfer_function_str)
        self.unit = OnUnit(spatiotemporal_filter, transfer_function)


class LGNOffCell(OffUnit):
    def __init__(self, **kwargs):
        lattice_unit_center = kwargs.pop('lattice_unit_center', None)
        weights = kwargs.pop('weights', None)
        kpeaks = kwargs.pop('kpeaks', None)
        amplitude = kwargs.pop('amplitude', None)
        sigma = kwargs.pop('sigma', None)
        width = kwargs.pop('width', 5)
        transfer_function_str = kwargs.pop('transfer_function_str', 'Heaviside(s)*s')

        dxi = np.random.uniform(-width*1./2,width*1./2)
        dyi = np.random.uniform(-width*1./2,width*1./2)
        temporal_filter = TemporalFilterCosineBump(weights, kpeaks)
        spatial_filter = GaussianSpatialFilter(translate=(dxi,dyi), sigma=sigma,  origin=lattice_unit_center) # all distances measured from BOTTOM LEFT
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
        transfer_function = ScalarTransferFunction(transfer_function_str)
        super(LGNOnCell, self).__init__(spatiotemporal_filter, transfer_function)


if __name__ == "__main__":
    movie_file = '/data/mat/iSee_temp_shared/movies/TouchOfEvil.npy'
    m_data = np.load(movie_file, 'r')
    m = Movie(m_data[1000:], frame_rate=30.)
    
    # Create second cell:
    transfer_function = ScalarTransferFunction('s')
    temporal_filter = TemporalFilterCosineBump((.4,-.3), (20,60))
    cell_list = []
    for xi in np.linspace(0,m.data.shape[2], 5):
        for yi in np.linspace(0,m.data.shape[1], 5):
            spatial_filter_on = GaussianSpatialFilter(sigma=(2,2), origin=(0,0), translate=(xi, yi))
            on_linear_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter, amplitude=20)
            spatial_filter_off = GaussianSpatialFilter(sigma=(4,4), origin=(0,0), translate=(xi, yi))
            off_linear_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter, amplitude=-20) 
            on_off_cell = LGNOnOffCell(on_linear_filter, off_linear_filter)
            cell_list.append(on_off_cell)
    
    lgn = LGNModel(cell_list) #Here include a list of all cells
    y = lgn.evaluate(m, downsample=100) #Does the filtering + non-linearity on movie object m
    heat_plot(y, interpolation='none', colorbar=True)
      



 
# 
#     def imshow(self, ii, image_shape, fps, ax=None, show=True, relative_spatial_location=(0,0)):
#         
#         if ax is None:
#             _, ax = plt.subplots(1,1)
#         
#         curr_kernel = self.get_spatio_temporal_kernel(image_shape, fps, relative_spatial_location=relative_spatial_location)
#         
#         cax = curr_kernel.imshow(ii, ax=ax, show=False)
#         
#         if show == True:
#             plt.show()
#         
#         return ax
# 
# 
# class OnOffCellModel(CellModel):
#      
#     def __init__(self, dc_offset=0, on_subfield=None, off_subfield=None, on_weight = 1, off_weight = -1, t_max=None):
#  
#         super(self.__class__, self).__init__(dc_offset, t_max)
#  
#         if isinstance(on_subfield, dict):
#             curr_module, curr_class = on_subfield.pop('class')
#             self.on_subfield = getattr(importlib.import_module(curr_module), curr_class)(**on_subfield)
#         else:
#             self.on_subfield = on_subfield
#             
#         super(self.__class__, self).add_subfield(on_subfield, on_weight)
#             
#         if isinstance(off_subfield, dict):
#             curr_module, curr_class = off_subfield.pop('class')
#             self.off_subfield = getattr(importlib.import_module(curr_module), curr_class)(**off_subfield)
#         else:
#             self.off_subfield = off_subfield
#  
#         super(self.__class__, self).add_subfield(off_subfield, off_weight)
#  
#  
#     def to_dict(self):
#         
#         return {'dc_offset':self.dc_offset,
#                 'on_subfield':self.on_subfield.to_dict(),
#                 'off_subfield':self.off_subfield.to_dict(),
#                 't_max':self.t_max,
#                 'class':(__name__, self.__class__.__name__)}
#         
# class SingleSubfieldCellModel(CellModel):
#      
#     def __init__(self, subfield, weight = 1, dc_offset=0, t_max=None):
#  
#         super(SingleSubfieldCellModel, self).__init__(dc_offset, t_max)
#  
#         if isinstance(subfield, dict):
#             curr_module, curr_class = subfield.pop('class')
#             subfield = getattr(importlib.import_module(curr_module), curr_class)(**subfield)
#             
#         super(self.__class__, self).add_subfield(subfield, weight)
#  
#     def to_dict(self):
#         
#         assert len(self.subfield_list) == 1
#         subfield = self.subfield_list[0]
#         weight = self.subfield_weight_dict[subfield]
#         
#         return {'dc_offset':self.dc_offset,
#                 'subfield':subfield.to_dict(),
#                 'weight':weight,
#                 't_max':self.t_max,
#                 'class':(__name__, self.__class__.__name__)}
#         
# class OnCellModel(SingleSubfieldCellModel):
#     
#     def __init__(self, on_subfield, weight = 1, dc_offset=0 , t_max=None):
#         assert weight > 0
#         super(OnCellModel, self).__init__(on_subfield, weight, dc_offset, t_max)
#         
#     def to_dict(self):
#         data_dict = super(OnCellModel, self).to_dict()
#         data_dict['on_subfield'] = data_dict.pop('subfield')
#         return data_dict
#         
# class OffCellModel(SingleSubfieldCellModel):
#     
#     def __init__(self, on_subfield, weight = -1, dc_offset=0 , t_max=None):
#         assert weight < 0
#         super(OffCellModel, self).__init__(on_subfield, weight, dc_offset, t_max)
#         
#     def to_dict(self):
#         data_dict = super(OffCellModel, self).to_dict()
#         data_dict['off_subfield'] = data_dict.pop('subfield')
#         return data_dict
        
        
# class OffCellModel(CellModel):
#      
#     def __init__(self, off_subfield, dc_offset=0, off_weight = 1, t_max=None):
#         
#         assert off_weight < 0.
#         self.weight = off_weight
#         
# 
#         
#  
#         super(self.__class__, self).__init__(dc_offset, t_max)
#  
#         if isinstance(on_subfield, dict):
#             curr_module, curr_class = on_subfield.pop('class')
#             self.subfield = getattr(importlib.import_module(curr_module), curr_class)(**on_subfield)
#         else:
#             self.subfield = on_subfield
#             
#         super(self.__class__, self).add_subfield(self.subfield, self.weight)
#  
#     def to_dict(self):
#         
#         return {'dc_offset':self.dc_offset,
#                 'on_subfield':self.subfield.to_dict(),
#                 'on_weight':self.weight,
#                 't_max':self.t_max,
#                 'class':(__name__, self.__class__.__name__)}


 


        
# if __name__ == "__main__":
#     
#     t = np.arange(0,.5,.001)
#     example_movie = movie.Movie(file_name=os.path.join(isee_engine.movie_directory, 'TouchOfEvil.npy'), frame_rate=30.1, memmap=True)
#     
#     temporal_filter_on = TemporalFilterExponential(weight=1, tau=.05)
#     on_subfield = Subfield(scale=(5,15), weight=.5, rotation=30, temporal_filter=temporal_filter_on, translation=(0,0))
#     
#     temporal_filter_off = TemporalFilterExponential(weight=2, tau=.01)
#     off_subfield = Subfield(scale=(5,15), weight=.5, rotation=-30, temporal_filter=temporal_filter_off)
# 
#     cell = OnOffCellModel(on_subfield=on_subfield, off_subfield=off_subfield, dc_offset=0., t_max=.5)
#     curr_kernel = cell.get_spatio_temporal_kernel((100,150), 30.1)
#     curr_kernel.imshow(0)
#     
#     print cell.to_dict()
 


#     f = cell.get_spatio_temporal_filter(example_movie.movie_data.shape[1:], t,threshold=.5)
#     print len(f.t_ind_list)
#     
#     
    
#     for ii in range(example_movie.number_of_frames-curr_filter.t_max):
#         print ii, example_movie.number_of_frames, curr_filter.map(example_movie, ii)


#     off_subfield = Subfield(scale=(15,15), weight=.2, translation=(30,30))


#     
#     curr_filter = cell.get_spatio_temporal_filter((100,150))
#     

#     
# #     print touch_of_evil(40.41, mask=m)
#     print curr_filter.t_max
#     for ii in range(example_movie.number_of_frames-curr_filter.t_max):
#         print ii, example_movie.number_of_frames, curr_filter.map(example_movie, ii)
    
#     cell.visualize_spatial_filter((100,150))
#         show_volume(spatio_temporal_filter, vmin=spatio_temporal_filter.min(), vmax=spatio_temporal_filter.max())
    


#     def get_spatial_filter(self, image_shape, relative_spatial_location=(0,0), relative_threshold=default_relative_threshold):
#         
#         # Initialize:
#         translation_matrix = util.get_translation_matrix(relative_spatial_location)
#         
#         # On-subunit:
#         on_filter_pre_spatial = self.on_subfield.get_spatial_filter(image_shape)
#         on_filter_spatial = util.apply_transformation_matrix(on_filter_pre_spatial, translation_matrix)
#         
#         # Off-subunit:
#         off_filter_pre_spatial = self.off_subfield.get_spatial_filter(image_shape)
#         off_filter_spatial = util.apply_transformation_matrix(off_filter_pre_spatial, translation_matrix)
# 
#         spatial_filter = on_filter_spatial - off_filter_spatial
# 
#         tmp = np.abs(spatial_filter)
#         spatial_filter[np.where(tmp/tmp.max() < relative_threshold )] = 0
# 
#         return spatial_filter

# kernel = float(self.dc_offset)/len(nonzero_ind_tuple[0])+spatio_temporal_filter[nonzero_ind_tuple]

# def rectifying_filter_factory(kernel, movie, dc_offset=0):
#     
#     def rectifying_filter(t):
#         
#         fi = movie.frame_rate*float(t)
#         fim, fiM = np.floor(fi), np.ceil(fi)
#         
#         print t, fim, fiM
#         
#         try:
#             s1 = (movie.movie_data[int(fim)+kernel.t_ind_list, kernel.row_ind_list, kernel.col_ind_list]*kernel.kernel).sum()
#             s2 = (movie.movie_data[int(fiM)+kernel.t_ind_list, kernel.row_ind_list, kernel.col_ind_list]*kernel.kernel).sum()
#         except IndexError:
#             return None
#         
#         # Linear interpolation:
#         s_pre = dc_offset + s1*((1-(fi-fim))*.5) + s2*((fi-fim)*.5)
#         
#         if s_pre < 0:
#             return 0
#         else:
#             return float(s_pre)
#     
#     return rectifying_filter

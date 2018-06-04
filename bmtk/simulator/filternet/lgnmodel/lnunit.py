import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from . import utilities as util
import importlib
from .kernel import Kernel2D, Kernel3D
from .linearfilter import SpatioTemporalFilter
import json
from .spatialfilter import GaussianSpatialFilter
from .transferfunction import ScalarTransferFunction
from .temporalfilter import TemporalFilterCosineBump
from .cursor import LNUnitCursor, MultiLNUnitCursor, MultiLNUnitMultiMovieCursor, SeparableLNUnitCursor, SeparableMultiLNUnitCursor
from .movie import Movie
from .lgnmodel1 import LGNModel, heat_plot
from .transferfunction import MultiTransferFunction, ScalarTransferFunction
    
    
class LNUnit(object):
     
    def __init__(self, linear_filter, transfer_function, amplitude=1.):
        
        self.linear_filter = linear_filter
        self.transfer_function = transfer_function
        self.amplitude = amplitude

    def evaluate(self, movie, **kwargs):
        return self.get_cursor(movie, separable=kwargs.pop('separable', False)).evaluate(**kwargs)
 
    def get_spatiotemporal_kernel(self, *args, **kwargs):
        return self.linear_filter.get_spatiotemporal_kernel(*args, **kwargs)
    
    def get_cursor(self, movie, threshold=0, separable = False):
        if separable:
            return SeparableLNUnitCursor(self, movie)
        else:
            return LNUnitCursor(self, movie, threshold=threshold)
    
    def show_temporal_filter(self, *args, **kwargs):
        self.linear_filter.show_temporal_filter(*args, **kwargs)
        
    def show_spatial_filter(self, *args, **kwargs):
        self.linear_filter.show_spatial_filter(*args, **kwargs)
    
    def to_dict(self):
        return {'class':(__name__, self.__class__.__name__),
                'linear_filter':self.linear_filter.to_dict(),
                'transfer_function':self.transfer_function.to_dict()}
        
class MultiLNUnit(object):
    
    def __init__(self, lnunit_list, transfer_function):
        
        self.lnunit_list = lnunit_list
        self.transfer_function = transfer_function
        
    def get_spatiotemporal_kernel(self, *args, **kwargs):
        
        k = Kernel3D([],[],[],[],[],[],[])
        for unit in self.lnunit_list:
            k = k+unit.get_spatiotemporal_kernel(*args, **kwargs)
            
        return k
        
    def show_temporal_filter(self, *args, **kwargs):
        
        ax = kwargs.pop('ax', None)
        show = kwargs.pop('show', None)
        save_file_name = kwargs.pop('save_file_name', None) 
        
        
        if ax is None:
            _, ax = plt.subplots(1,1)
        
        kwargs.update({'ax':ax, 'show':False, 'save_file_name':None})
        for unit in self.lnunit_list:
            if unit.linear_filter.amplitude < 0:
                color='b'
            else:
                color='r'
            unit.linear_filter.show_temporal_filter(color=color, **kwargs)

        if not save_file_name is None:
            plt.savefig(save_file_name, transparent=True)
         
        if show == True:
            plt.show()
         
        return ax
    
    def show_spatial_filter(self, *args, **kwargs):
        
        ax = kwargs.pop('ax', None)
        show = kwargs.pop('show', True)
        save_file_name = kwargs.pop('save_file_name', None) 
        colorbar = kwargs.pop('colorbar', True) 
        
        k = Kernel2D(args[0],args[1],[],[],[])
        for lnunit in self.lnunit_list:
            k = k + lnunit.linear_filter.spatial_filter.get_kernel(*args, **kwargs)
        k.imshow(ax=ax, show=show, save_file_name=save_file_name, colorbar=colorbar)
        
    def get_cursor(self, *args, **kwargs):
        
        threshold = kwargs.get('threshold', 0.)
        separable = kwargs.get('separable', False)
        
        if len(args) == 1:
            movie = args[0]
            if separable:
                return SeparableMultiLNUnitCursor(self, movie)
            else:
                return MultiLNUnitCursor(self, movie, threshold=threshold)
        elif len(args) > 1:
            movie_list = args
            if separable:
                raise NotImplementedError
            else:
                return MultiLNUnitMultiMovieCursor(self, movie_list, threshold=threshold)
        else:
            assert ValueError
    
    
    def evaluate(self, movie, **kwargs):
        seperable = kwargs.pop('separable', False)
        return self.get_cursor(movie, separable=seperable).evaluate(**kwargs)

from sympy.abc import x, y

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
            on_lnunit = LNUnit(on_linear_filter, transfer_function)
            spatial_filter_off = GaussianSpatialFilter(sigma=(4,4), origin=(0,0), translate=(xi, yi))
            off_linear_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter, amplitude=-20)        
            off_lnunit = LNUnit(off_linear_filter, transfer_function)
            
            multi_transfer_function = MultiTransferFunction((x, y), 'x+y')
            
            multi_unit = MultiLNUnit([on_lnunit, off_lnunit], multi_transfer_function)
            cell_list.append(multi_unit)
    
    lgn = LGNModel(cell_list) #Here include a list of all cells
    y = lgn.evaluate(m, downsample=10) #Does the filtering + non-linearity on movie object m
    heat_plot(y, interpolation='none', colorbar=False)
      



 
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
import numpy as np
from kernel import Kernel3D
import matplotlib.pyplot as plt

class SpatioTemporalFilter(object):

    def __init__(self, spatial_filter, temporal_filter, amplitude=1.):
        
        self.spatial_filter  = spatial_filter
        self.temporal_filter = temporal_filter
        self.amplitude = amplitude

    def get_spatiotemporal_kernel(self, row_range, col_range, t_range=None, threshold=0, reverse=False):

        spatial_kernel = self.spatial_filter.get_kernel(row_range, col_range, threshold=0)
        temporal_kernel = self.temporal_filter.get_kernel(t_range=t_range, threshold=0, reverse=reverse)

        t_range = temporal_kernel.t_range
                
        spatiotemporal_kernel = np.ones(( len(temporal_kernel), len(spatial_kernel)))
        spatiotemporal_kernel *= spatial_kernel.kernel[None, :]

        spatiotemporal_kernel *= temporal_kernel.kernel[:,None]
        spatiotemporal_kernel = spatiotemporal_kernel.reshape((np.prod(spatiotemporal_kernel.shape)))
        
        spatial_coord_array = np.empty((len(spatial_kernel),2))
        spatial_coord_array[:,0] = spatial_kernel.col_inds
        spatial_coord_array[:,1] = spatial_kernel.row_inds
         
        spatiiotemporal_coord_array = np.zeros((len(spatial_kernel)*len(temporal_kernel),3))
        spatiiotemporal_coord_array[:,0:2] = np.kron(np.ones((len(temporal_kernel),1)),spatial_coord_array)
        spatiiotemporal_coord_array[:,2] = np.kron(temporal_kernel.t_inds, np.ones(len(spatial_kernel)))
        
        col_inds, row_inds, t_inds = map(lambda x:x.astype(np.int),spatiiotemporal_coord_array.T)
        kernel = Kernel3D(spatial_kernel.row_range, spatial_kernel.col_range, t_range, row_inds, col_inds, t_inds, spatiotemporal_kernel)
        kernel.apply_threshold(threshold)
        

        kernel.kernel *= self.amplitude

        
        return kernel
    
    def t_slice(self, t, *args, **kwargs):
        
        k = self.get_spatiotemporal_kernel(*args, **kwargs)
        return k.t_slice(t)
    
    def show_temporal_filter(self, *args, **kwargs):

        self.temporal_filter.imshow(*args, **kwargs)
            
    def show_spatial_filter(self, *args, **kwargs):
        
        self.spatial_filter.imshow(*args, **kwargs)
    
    def to_dict(self):
        
        return {'class':(__name__, self.__class__.__name__),
                'spatial_filter':self.spatial_filter.to_dict(),
                'temporal_filter':self.temporal_filter.to_dict(),
                'amplitude':self.amplitude}

# class OnOffSpatioTemporalFilter(SpatioTemporalFilter):
# 
#     def __init__(self, on_spatiotemporal_filter, off_spatiotemporal_filter):
#         
#         self.on_spatiotemporal_filter = on_spatiotemporal_filter
#         self.off_spatiotemporal_filter = off_spatiotemporal_filter
# 
#     def get_spatiotemporal_kernel(self, col_range, row_range, t_range=None, threshold=0, reverse=False):
#         
#         on_kernel = self.on_spatiotemporal_filter.get_spatiotemporal_kernel(col_range, row_range, t_range, threshold, reverse)
#         off_kernel = self.off_spatiotemporal_filter.get_spatiotemporal_kernel(col_range, row_range, t_range, threshold, reverse)
# 
#         return on_kernel + off_kernel*(-1)
# 
#     def to_dict(self):
#         
#         return {'class':(__name__, self.__class__.__name__),
#                 'on_filter':self.on_spatiotemporal_filter.to_dict(),
#                 'off_filter':self.off_spatiotemporal_filter.to_dict()}
#         
# class TwoSubfieldLinearFilter(OnOffSpatioTemporalFilter):
# 
#     def __init__(self, dominant_spatiotemporal_filter, nondominant_spatiotemporal_filter, subfield_separation=10, onoff_axis_angle=45, dominant_subfield_location=(30,40)):
#         
#         self.subfield_separation = subfield_separation
#         self.onoff_axis_angle = onoff_axis_angle
#         self.dominant_subfield_location = dominant_subfield_location
#         self.dominant_spatiotemporal_filter = dominant_spatiotemporal_filter
#         self.nondominant_spatiotemporal_filter = nondominant_spatiotemporal_filter
#         
#         dom_amp = dominant_spatiotemporal_filter.spatial_filter.amplitude
#         nondom_amp = nondominant_spatiotemporal_filter.spatial_filter.amplitude
#         if dom_amp < 0 and nondom_amp > 0:
#             super(TwoSubfieldLinearFilter, self).__init__(self.nondominant_spatiotemporal_filter, self.dominant_spatiotemporal_filter)
#         elif dom_amp > 0 and nondom_amp < 0:    
#             super(TwoSubfieldLinearFilter, self).__init__(self.dominant_spatiotemporal_filter, self.nondominant_spatiotemporal_filter)
#         else:
#             raise ValueError('Subfields are not of opposite polarity')
#         
#         self.dominant_spatiotemporal_filter.spatial_filter.translate = self.dominant_subfield_location
#         hor_offset = np.cos(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation + self.dominant_subfield_location[0]
#         vert_offset = np.sin(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation+ self.dominant_subfield_location[1]
#         rel_translation = (hor_offset,vert_offset)
#         self.nondominant_spatiotemporal_filter.spatial_filter.translate = rel_translation
#         self.nondominant_spatiotemporal_filter.spatial_filter.origin=self.dominant_spatiotemporal_filter.spatial_filter.origin
#         
#         
#     def to_dict(self):
#         
#         raise NotImplementedError
#         
        
        
        
        
        
        
        
        
        
        
        
        
        
        

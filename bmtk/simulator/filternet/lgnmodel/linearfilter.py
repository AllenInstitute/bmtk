import numpy as np

from .kernel import Kernel3D


class SpatioTemporalFilter(object):
    def __init__(self, spatial_filter, temporal_filter, amplitude=1.):
        self.spatial_filter = spatial_filter
        self.temporal_filter = temporal_filter
        self.amplitude = amplitude

    def get_spatiotemporal_kernel(self, row_range, col_range, t_range=None, threshold=0, reverse=False):
        # TODO: Rename to get_kernel() to match with spatialfilter and temporalfilter
        spatial_kernel = self.spatial_filter.get_kernel(row_range, col_range, threshold=0)
        temporal_kernel = self.temporal_filter.get_kernel(t_range=t_range, threshold=0, reverse=reverse)

        t_range = temporal_kernel.t_range
                
        spatiotemporal_kernel = np.ones((len(temporal_kernel), len(spatial_kernel)))
        spatiotemporal_kernel *= spatial_kernel.kernel[None, :]

        spatiotemporal_kernel *= temporal_kernel.kernel[:, None]
        spatiotemporal_kernel = spatiotemporal_kernel.reshape((np.prod(spatiotemporal_kernel.shape)))
        
        spatial_coord_array = np.empty((len(spatial_kernel), 2))
        spatial_coord_array[:, 0] = spatial_kernel.col_inds
        spatial_coord_array[:, 1] = spatial_kernel.row_inds

        spatiiotemporal_coord_array = np.zeros((len(spatial_kernel)*len(temporal_kernel), 3))
        spatiiotemporal_coord_array[:, 0:2] = np.kron(np.ones((len(temporal_kernel), 1)), spatial_coord_array)
        spatiiotemporal_coord_array[:, 2] = np.kron(temporal_kernel.t_inds, np.ones(len(spatial_kernel)))
        
        col_inds, row_inds, t_inds = map(lambda x: x.astype(np.int), spatiiotemporal_coord_array.T)
        kernel = Kernel3D(spatial_kernel.row_range, spatial_kernel.col_range, t_range, row_inds, col_inds, t_inds,
                          spatiotemporal_kernel)
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
        return {'class': (__name__, self.__class__.__name__),
                'spatial_filter': self.spatial_filter.to_dict(),
                'temporal_filter': self.temporal_filter.to_dict(),
                'amplitude': self.amplitude}

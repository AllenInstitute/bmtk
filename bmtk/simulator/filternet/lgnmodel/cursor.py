import numpy as np
import scipy.signal as spsig

from .utilities import convert_tmin_tmax_framerate_to_trange


class KernelCursor(object):
    """A class that takes care of the convolution of the (non-separable) spatial-temporal linear filter with the move.
    """
    def __init__(self, kernel, movie):
        self.movie = movie
        self.kernel = kernel
        self.cache = {}

        # This ensures that the kernel frame rate matches the movie frame rate:
        np.testing.assert_almost_equal(np.diff(self.kernel.t_range),
                                       np.ones_like(self.kernel.t_range[1:])*(1./movie.frame_rate))
        
    @property
    def row_range(self):
        return self.movie.row_range
        
    @property
    def col_range(self):
        return self.movie.col_range
    
    @property
    def t_range(self):
        return self.movie.t_range
    
    @property
    def frame_rate(self):
        return self.movie.frame_rate
    
    def evaluate(self, t_min=None, t_max=None, downsample=1):
        if t_max is None:
            t_max = self.t_range[-1]
            
        if t_min is None:
            t_min = self.t_range[0]
        
        t_range = convert_tmin_tmax_framerate_to_trange(t_min, t_max, self.movie.frame_rate)[::int(downsample)]
        y_vals = np.array([self(t) for t in t_range])

        return t_range, y_vals  
    
    def __call__(self, t):
        # TODO: Using call is not a good idea here, change to evaluate
        if t < self.t_range[0] or t > self.t_range[-1]:
            curr_rate = 0
        else:
            ti = t*self.frame_rate
            til, tir = int(np.floor(ti)), int(np.ceil(ti))
            
            tl, tr = float(til)/self.frame_rate, float(tir)/self.frame_rate
            if np.abs(tl - t) < 1e-12:
                curr_rate = self.apply_dot_product(til)

            elif np.abs(tr - t) < 1e-12:
                curr_rate = self.apply_dot_product(tir)

            else:
                wa, wb = (1-(t-tl)/(tr-tl)), (1-(tr-t)/(tr-tl))
                cl = self.apply_dot_product(til)
                cr = self.apply_dot_product(tir)             
                curr_rate = cl*wa+cr*wb

        if np.isnan(curr_rate):
            assert RuntimeError
        
        return curr_rate

    def apply_dot_product(self, ti_offset):
        try:
            # TODO: This needs to be altered asap
            return self.cache[ti_offset]
        
        except KeyError:
            t_inds = self.kernel.t_inds + ti_offset + 1  # Offset by one nhc 14 Apr '17
            min_ind, max_ind = 0, self.movie.data.shape[0]
            allowed_inds = np.where(np.logical_and(min_ind <= t_inds, t_inds < max_ind))
            t_inds = t_inds[allowed_inds]
            row_inds = self.kernel.row_inds[allowed_inds]
            col_inds = self.kernel.col_inds[allowed_inds]
            kernel_vector = self.kernel.kernel[allowed_inds] 
            result = np.dot(self.movie[t_inds, row_inds, col_inds], kernel_vector)
            self.cache[ti_offset] = result
            return result


class FilterCursor(KernelCursor):
    def __init__(self, spatiotemporal_filter, movie, threshold=0):
        # TODO: not sure why this needs to have it's own class and shouldn't be merged into parent?
        self.spatiotemporal_filter = spatiotemporal_filter
        kernel = self.spatiotemporal_filter.get_spatiotemporal_kernel(movie.row_range, movie.col_range,
                                                                      t_range=movie.t_range, threshold=threshold,
                                                                      reverse=True)

        super(FilterCursor, self).__init__(kernel, movie)


class LNUnitCursor(KernelCursor):
    """A class that takes care of applying a linear-nonlinear filter to a movie. Parent class is used to apply the
    spatial-termporal filter convolution to the movie, when then the lnunit non-linear transfer function is applied.

    """
    def __init__(self, lnunit, movie, threshold=0):
        self.lnunit = lnunit
        kernel = lnunit.get_spatiotemporal_kernel(movie.row_range, movie.col_range, movie.t_range, reverse=True,
                                                  threshold=threshold)
        kernel.apply_threshold(threshold)
             
        super(LNUnitCursor, self).__init__(kernel, movie)
    
    def __call__(self, t):
        # TODO: Don't use call operator, change to evaluate
        return self.lnunit.transfer_function(super(LNUnitCursor, self).__call__(t))


class MultiLNUnitCursor(object):
    def __init__(self, multi_lnunit, movie, threshold=0):
        self.multi_lnunit = multi_lnunit
        self.lnunit_cursor_list = [LNUnitCursor(lnunit, movie, threshold=threshold)
                                   for lnunit in multi_lnunit.lnunit_list]
        self.movie = movie
        
    def evaluate(self, **kwargs):
        multi_e = [unit_cursor.evaluate(**kwargs) for unit_cursor in self.lnunit_cursor_list]
        t_list, y_list = zip(*multi_e)
        return t_list[0], self.multi_lnunit.transfer_function(*y_list)


class MultiLNUnitMultiMovieCursor(MultiLNUnitCursor):
    def __init__(self, multi_lnunit, movie_list, threshold=0.):
        assert len(multi_lnunit.lnunit_list) == len(movie_list)
        
        self.multi_lnunit = multi_lnunit
        self.lnunit_movie_list = movie_list
        self.lnunit_cursor_list = [lnunit.get_cursor(movie, threshold=threshold) for
                                   lnunit, movie in zip(multi_lnunit.lnunit_list, movie_list)]


class SeparableKernelCursor(object):
    """A class for applying a spatial-temporal convolution to a movie. Unlike the KernelCursor the spatial-temporal
    filter is broken up into its components, spatial filter is applied then followed by a temporal filter convolution.

    """
    def __init__(self, spatial_kernel, temporal_kernel, movie):
        """Assumes temporal kernel is not reversed"""

        self.movie = movie
        self.spatial_kernel = spatial_kernel
        self.temporal_kernel = temporal_kernel

    def evaluate(self, threshold=0):
        full_spatial_kernel = np.array([self.spatial_kernel.full()])
        full_temporal_kernel = self.temporal_kernel.full()

        # Convolve every frame in the movie with the spatial filter. First find the range of rows (range min and max)
        #  and columns in the filter that are above threshold, that way only portion of movie/filter are multiplied
        # together
        nonzero_inds = np.where(np.abs(full_spatial_kernel[0, :, :]) >= threshold)
        rm, rM = nonzero_inds[0].min(), nonzero_inds[0].max()
        cm, cM = nonzero_inds[1].min(), nonzero_inds[1].max()
        convolution_answer_sep_spatial = (self.movie.data[:, rm:rM+1, cm:cM+1] *
                                          full_spatial_kernel[:, rm:rM+1, cm:cM+1]).sum(axis=1).sum(axis=1)

        # Convolve results of spatial convolution with the temporal filter
        sig_tmp = np.zeros(len(full_temporal_kernel) + len(convolution_answer_sep_spatial) - 1)
        sig_tmp[len(full_temporal_kernel)-1:] = convolution_answer_sep_spatial
        convolution_answer_sep = spsig.convolve(sig_tmp, full_temporal_kernel[::-1], mode='valid')
        t = np.arange(len(convolution_answer_sep))/self.movie.frame_rate

        return t, convolution_answer_sep


class SeparableSpatioTemporalFilterCursor(SeparableKernelCursor):
    def __init__(self, spatiotemporal_filter, movie):
        self.spatial_filter = spatiotemporal_filter.spatial_filter
        self.temporal_filter = spatiotemporal_filter.temporal_filter

        spatial_kernel = self.spatial_filter.get_kernel(movie.row_range, movie.col_range, threshold=-1)
        temporal_kernel = self.temporal_filter.get_kernel(t_range=movie.t_range, threshold=0, reverse=True)
        spatial_kernel.kernel *= spatiotemporal_filter.amplitude

        super(SeparableSpatioTemporalFilterCursor, self).__init__(spatial_kernel, temporal_kernel, movie)


class SeparableLNUnitCursor(SeparableSpatioTemporalFilterCursor):
    def __init__(self, lnunit, movie):
        self.lnunit = lnunit
        super(SeparableLNUnitCursor, self).__init__(self.lnunit.linear_filter, movie)

    def evaluate(self, downsample=1):
        assert(downsample == 1)
        t, y = super(SeparableLNUnitCursor, self).evaluate()
        return t, [self.lnunit.transfer_function(yi) for yi in y]


class SeparableMultiLNUnitCursor(object):
    def __init__(self, multilnunit, movie):
        self.multilnunit = multilnunit
        self.lnunit_cursor_list = []
        for lnunit in self.multilnunit.lnunit_list:
            self.lnunit_cursor_list.append(SeparableLNUnitCursor(lnunit, movie))

    def evaluate(self, *args, **kwargs):
        assert(kwargs.get('downsample', 1) == 1)
        y_list = []
        for cursor in self.lnunit_cursor_list:
            t, y = cursor.evaluate(*args, **kwargs)
            y_list.append(y)

        return t, self.multilnunit.transfer_function(*y_list)

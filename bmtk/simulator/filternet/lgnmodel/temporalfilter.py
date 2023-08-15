import numpy as np
import scipy.interpolate as spinterp

from . import fitfuns
from .kernel import Kernel1D


class TemporalFilter(object):
    def __init__(self):
        self.t_support = []
        self.kernel_data = None

    def imshow(self, t_range=None, threshold=0, reverse=False, rescale=False, **kwargs):
        return self.get_kernel(t_range, threshold, reverse, rescale).imshow(**kwargs)
        
    def to_dict(self):
        return {'class': (__name__, self.__class__.__name__)}

    def get_default_t_grid(self):
        raise NotImplementedError()

    def get_kernel(self, t_range=None, threshold=0, reverse=False, rescale=False):
        if t_range is None:
            t_range = self.get_default_t_grid() 
        
        if len(self.t_support) == 1:
            k = Kernel1D(self.t_support, self.kernel_data, threshold=threshold, reverse=reverse)

        else:
            interpolation_function = spinterp.interp1d(self.t_support, self.kernel_data, fill_value=0,
                                                       bounds_error=False, assume_sorted=True)
            k = Kernel1D(t_range, interpolation_function(t_range), threshold=threshold, reverse=reverse)

        if rescale:
            k.rescale()
            assert(np.abs(np.abs(k.kernel.sum()) - 1) < 1e-14)
        
        return k


class ArrayTemporalFilter(TemporalFilter):
    def __init__(self, mask, t_support):
        super(ArrayTemporalFilter, self).__init__()
        self.mask = mask
        self.t_support = t_support
        assert(len(self.mask) == len(self.t_support))
        self.nkt = 600
        self.kernel_data = self.mask

    def get_default_t_grid(self):
        return np.arange(self.nkt)*0.001


class TemporalFilterCosineBump(TemporalFilter):
    def __init__(self, weights, kpeaks, delays):
        """Creates a time-based filter function by combining two cosine-based peaks into a function for convoluting
        the input with.

        :param weights: (float, float)the magntiude of both peaks, first weight must be positive
        :param kpeaks: (float, float) the spread of each peak, first peak (0) must be sharper
        :param delays: (float, float) the delay of both peaks, peak 0 must be positive occur first.
        """

        assert(len(kpeaks) == 2)
        assert(kpeaks[0] < kpeaks[1])
        assert(weights[0] > 0)
        assert(delays[0] <= delays[1])

        super(TemporalFilterCosineBump, self).__init__()
        self.ncos = len(weights)

        # Not likely to change defaults:
        self.neye = 0
        self.b = .3
        self.nkt = 600

        # Parameters
        self.weights = np.array([weights]).T
        self.kpeaks = kpeaks
        self.delays = np.array([delays]).astype(int)

        # Create two peak arrays (arr0 and arr1) using makeBasisStimKernel. Then merge them using dot product
        # w0*arr0 + w1*arr1.
        kbasprs = {
            'neye': self.neye,
            'ncos': self.ncos,
            'kpeaks': self.kpeaks,
            'b': self.b,
            'delays': self.delays
        }
        nkt = self.nkt
        self.kernel_data = np.dot(fitfuns.makeBasis_StimKernel(kbasprs, nkt), self.weights)[::-1].T[0]

        self.t_support = np.arange(0, len(self.kernel_data)*.001, .001)
        self.kbasprs = kbasprs
        assert len(self.t_support) == len(self.kernel_data)
        
    def __call__(self, t):
        return self.interpolation_function(t)
        
    def get_default_t_grid(self):
        return np.arange(self.nkt)*.001
    
    def to_dict(self):
        param_dict = super(TemporalFilterCosineBump, self).to_dict()
        param_dict.update({'weights': self.weights.tolist(), 'kpeaks': self.kpeaks})
        return param_dict

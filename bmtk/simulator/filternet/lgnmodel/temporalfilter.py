import numpy as np
from . import fitfuns
import scipy.interpolate as spinterp
import matplotlib.pyplot as plt
from .kernel import Kernel1D

class TemporalFilter(object):

    def __init__(self, *args, **kwargs): pass
    
    def imshow(self, t_range=None, threshold=0, reverse=False, rescale=False, **kwargs):
        return self.get_kernel(t_range, threshold, reverse, rescale).imshow(**kwargs)
        

    def to_dict(self):
        return {'class':(__name__, self.__class__.__name__)}
    
    def get_kernel(self, t_range=None, threshold=0, reverse=False, rescale=False):

        if t_range is None:
            t_range = self.get_default_t_grid() 
        
#         print self.t_support
#         print self.kernel_data
        
        if len(self.t_support) == 1:
            k = Kernel1D(self.t_support, self.kernel_data, threshold=threshold, reverse=reverse)
        else:
            interpolation_function = spinterp.interp1d(self.t_support, self.kernel_data, fill_value=0, bounds_error=False, assume_sorted=True)
            k = Kernel1D(t_range, interpolation_function(t_range), threshold=threshold, reverse=reverse)
        if rescale == True:
            k.rescale()
        
        #assert np.abs(np.abs(k.kernel).sum() - 1) < 1e-14
            assert np.abs(np.abs(k.kernel.sum()) - 1) < 1e-14
        
        return k

class ArrayTemporalFilter(TemporalFilter):
    
    def __init__(self, mask,t_support):
        
        self.mask = mask
        self.t_support = t_support
        
        assert len(self.mask) == len(self.t_support)
        
        self.nkt = 600
        
        super(self.__class__, self).__init__()
        
        self.kernel_data = self.mask
        #self.t_support = np.arange(0, len(self.kernel_data)*.001, .001)
        #assert len(self.t_support) == len(self.kernel_data) 
        
    def get_default_t_grid(self):
        
        return np.arange(self.nkt)*.001

class TemporalFilterCosineBump(TemporalFilter):
    
    def __init__(self, weights, kpeaks, delays):

        assert len(kpeaks) == 2
        assert kpeaks[0]<kpeaks[1]
        assert weights[0] > 0
        assert delays[0] <= delays[1]
        
        self.ncos = len(weights)

        # Not likely to change defaults:
        self.neye = 0
        self.b = .3
        self.nkt = 600

        super(self.__class__, self).__init__()
        
        # Parameters
        self.weights = np.array([weights]).T
        self.kpeaks = kpeaks
        self.delays = np.array([delays]).astype(int)

        # Adapter code to get filters from Ram's code:
        kbasprs = {}
        kbasprs['neye'] = self.neye
        kbasprs['ncos'] = self.ncos
        kbasprs['kpeaks'] = self.kpeaks
        kbasprs['b'] = self.b
        kbasprs['delays'] = self.delays
        nkt = self.nkt
        #kbasprs['bases'] = fitfuns.makeBasis_StimKernel(kbasprs, nkt)
        self.kernel_data = np.dot(fitfuns.makeBasis_StimKernel(kbasprs, nkt), self.weights)[::-1].T[0]
#        plt.figure()
#        plt.plot(self.kernel_data)
#        plt.show()
#        sys.exit()
        self.t_support = np.arange(0, len(self.kernel_data)*.001, .001)
        self.kbasprs = kbasprs
        assert len(self.t_support) == len(self.kernel_data)
        
    def __call__(self, t):
        return self.interpolation_function(t)
        
    def get_default_t_grid(self):
        return np.arange(self.nkt)*.001
    
    def to_dict(self):
        
        param_dict = super(self.__class__, self).to_dict()
        
        param_dict.update({'weights':self.weights.tolist(),
                           'kpeaks':self.kpeaks})
        
        return param_dict

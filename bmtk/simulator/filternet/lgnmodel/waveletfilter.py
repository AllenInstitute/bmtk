import ast
import numpy as np
import itertools
from six import string_types
from scipy import ndimage

#from . import utilities as util
from .kernel import Kernel2D

class WaveletFilter(object):
    def __init__(self, translate=4.0, sigma_f=1.0, b_t=1.0, order_t=3, theta=0.0, Lambda=1.0, psi=0.0, delay=0.0,
                 amplitude= 1., direction=1):
        """WaveletFilter: Sinusoidally modulated filter with Gaussian spectral envelope and gamma temporal envelope

        :param translate: float, the origin of the filter in units of log2(freq/min_freq)
        :param sigma_f: float, the gaussian std in direction of frequency (octaves)
        :param b_t: float, bandwidth for gammatone-like temporal envelope of filter
        :param order_t: int, order for gammatone-like temporal envelope of filter
        :param theta: float, rotation of the Gabor filter (radians)
        :param Lambda: float, wavelength of sinusoidal component
        :param psi: float, phase of sinusoidal component (radians)
        :param delay: float, additional temporal delay (in ms)
        :param amplitude: float, amplitude applied to filter after normalization
        :param direction: int, direction of tilt: +1 for upward, -1 for downward, 0 for no tilt
        """

        self.translate = translate
        self.sigma_f = sigma_f
        self.b_t = b_t
        self.order_t = order_t
        self.theta = theta
        self.Lambda = Lambda
        self.psi = psi
        self.delay = delay
        self.amplitude = amplitude
        self.direction = direction

    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range, threshold).imshow(**kwargs)
        
    def to_dict(self):
        return {
            'class': (__name__, self.__class__.__name__),
            'translate': self.translate,
            'sigma_f': self.sigma_f,
            'b_t': self.b_t,
            'order_t': self.order_t,
            'theta': self.theta,
            'lambda': self.Lambda,
            'psi': self.psi,
            'amplitude': self.amplitude,
            'delay': self.delay,
            'direction': self.direction
        }
        
    def get_kernel(self, row_range, col_range, threshold_rel=0.05, show_plot = False, plot_ax=None):
        """Creates a 2D gaussian filter (kernel) for the given dimensions which can be used

        :param row_range: array of times
        :param col_range: array of frequency centers
        :param threshold_rel: crop filter to region with amplitudes greater than threshold_rel * max value
        :param plot_ax: plot handle if plot of filter is desired
        :return: A Kernel2D object
        """
        (y, x) = np.meshgrid(row_range, col_range)

        if self.theta != np.pi/2:
            f_t = np.cos(self.theta) / self.Lambda
            translate_t = -0.1 * self.Lambda / np.cos(self.theta) + self.delay/1000
            env = (f_t * (x - translate_t)) ** (self.order_t - 1) * np.exp(-1 * self.b_t * f_t * (x - translate_t)) \
                  * np.exp(-.5 * (y - self.translate) ** 2 / self.sigma_f ** 2)
            env *= np.heaviside(x - translate_t, 0)  # Not valid for negative t
            wave = np.cos(2 * np.pi / self.Lambda * (
                        (x - translate_t) * np.cos(self.theta) +
                        self.direction*(y - self.translate) * np.sin(self.theta)) + self.psi)
        else:
            # Special case temporal modulation freq is 0, approximate a fast, mostly positive filter
            # b_t should be set to ~10 to achieve this fast, mostly positive filter
            # The step response adapts slightly to a flat steady-state
            f_t = 2.5
            translate_t = 0 + self.delay/1000
            env = (f_t * (x - translate_t)) ** (self.order_t - 1) * np.exp(-1 * self.b_t * f_t * (x - translate_t)) \
                  * np.sin(2*np.pi*f_t * (x - translate_t)) \
                  * np.exp(-.5 * (y - self.translate) ** 2 / self.sigma_f ** 2)
            env *= np.heaviside(x - translate_t,0)   # Not valid for negative t
            wave = np.cos(2 * np.pi / self.Lambda * (y - self.translate) + self.psi)

        filt = env * wave
        filt /= np.max(filt)

        threshold = threshold_rel * np.max(filt)
        kernel = Kernel2D.from_dense(row_range, col_range, filt, threshold=threshold)
        kernel.normalize2()     # Scale up large kernels which can hit low float limit when normalized
        kernel.kernel *= self.amplitude    # How do normalize and amplitude work together? seems like they would counteract each other?

        #kernel.imshow(ax = plot_ax, truncate_col=True, xlabel='Time(s)', ylabel='log(freq)')
        if show_plot:
            kernel.imshow(ax=plot_ax, truncate_col=False, colorbar=False, show=False)
        return kernel

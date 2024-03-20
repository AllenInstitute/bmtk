import ast
import numpy as np
import itertools
from six import string_types
from scipy import ndimage

#from . import utilities as util
from .kernel import Kernel2D

class GaborFilter(object):
    def __init__(self, translate=4.0, sigma_f=1.0, sigma_t=1.0, theta=0.0, Lambda=1.0, psi=0.0, amplitude= 1.,
                 direction=1):
        """Gabor-like filter: Sinusoidally modulated filter with Gaussian spectral and temporal envelope
        As currently implemented, the envelope is not rotated with the sinusoid.

        :param translate: float, the origin of the filter in units of log(freq/min_freq)
        :param sigma_f: float, the gaussian std in direction of frequency (octaves)
        :param sigma_t: float, the gaussian std in direction of time (s)
        :param theta: float, rotation of the Gabor filter (radians)
        :param Lambda: float, wavelength of sinusoidal component
        :param psi: float, phase of sinusoidal component (radians)
        :param amplitude: float, amplitude applied to filter after normalization
        :param direction: int, direction of tilt: +1 for upward, -1 for downward, 0 for no tilt
        """

        self.translate = translate
        self.sigma_f = sigma_f
        self.sigma_t = sigma_t
        self.theta = theta
        self.Lambda = Lambda
        self.psi = psi
        self.amplitude = amplitude
        self.direction = direction

    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range, threshold).imshow(**kwargs)
        
    def to_dict(self):
        return {
            'class': (__name__, self.__class__.__name__),
            'translate': self.translate,
            'sigma_f': self.sigma_f,
            'sigma_t': self.sigma_t,
            'theta': self.theta,
            'lambda': self.Lambda,
            'psi': self.psi,
            'amplitude': self.amplitude,
            'direction': self.direction
        }
        
    def get_kernel(self, row_range, col_range, threshold_rel=0.05):
        """Creates a 2D gaussian filter (kernel) for the given dimensions which can be used

        :param row_range: array of times
        :param col_range: array of frequency centers
        :param threshold_rel: crop filter to region with amplitudes greater than threshold_rel * max value
        :return: A Kernel2D object
        """
        (y, x) = np.meshgrid(row_range, col_range)

        translate_t = 0.5*self.Lambda*np.cos(self.theta)

        # Variant that rotates the envelope for a proper Gabor
        # sigma_f and sigma_t no longer strictly associated with each axis
        # gauss = np.exp(-.5 * (((x-translate_t) * np.cos(self.theta) + (y-self.translate) * np.sin(self.theta)) ** 2 / self.sigma_t ** 2
        #                      + (-(x-translate_t) * np.sin(self.theta) + (y-self.translate) * np.cos(self.theta)) ** 2 / self.sigma_f ** 2))

        gauss = np.exp(-.5 * ((x-translate_t)**2/self.sigma_t**2 + (y-self.translate)**2/self.sigma_f**2))

        wave = np.cos(2 * np.pi / self.Lambda * ((x-translate_t) * np.cos(self.theta) +
                                                 self.direction*(y-self.translate) * np.sin(self.theta)) + self.psi)
        gb = gauss * wave

        threshold = threshold_rel * np.max(gb)
        kernel = Kernel2D.from_dense(row_range, col_range, gb, threshold=threshold)
        kernel.normalize2()
        kernel.kernel *= self.amplitude
        # Uncomment this to visualize the filter
        #kernel.imshow(truncate_col=True, xlabel='Time(s)', ylabel='log(freq) re: 50 Hz')

        return kernel

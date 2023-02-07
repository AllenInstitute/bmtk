import ast
import numpy as np
import itertools
from six import string_types
from scipy import ndimage

#from . import utilities as util
from .kernel import Kernel2D

class GaborFilter(object):
    def __init__(self, translate=4.0, sigma_f=1.0, sigma_t=1.0, theta=0.0, Lambda=1.0, psi=0.0, amplitude= 1.):
        """Gabor-like filter

        :param translate: float, the center frequency of the gabor
        :param sigma_f: float, the gaussian std in direction of frequency
        :param sigma_t: float, the gaussian std in direction of time
        :param theta: float, rotation of the Gabor filter in radians
        :param Lambda: float, wavelength of sinusoidal
        :param psi: float, phase of sinusoidal factor relative to Gaussian in radians
        :param gamma: aspect ratio of 2D Gaussian
        """

        self.translate = translate

        '''
        if isinstance(sigma1, string_types):
            # TODO: Move this to calling method
            try:
                sigma = ast.literal_eval(sigma)
            except Exception as exc:
                pass
        '''

        self.sigma_f = sigma_f
        self.sigma_t = sigma_t
        self.theta = theta
        self.Lambda = Lambda
        self.psi = psi
        self.amplitude = amplitude

    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range, threshold).imshow(**kwargs)
        
    def to_dict(self):
        return {
            'class': (__name__, self.__class__.__name__),
            'translate': self.translate,
            'sigma1': self.sigma1,
            'sigma2': self.sigma2,
            'theta': self.theta,
            'lambda': self.Lambda,
            'psi': self.psi
        }
        
    def get_kernel(self, row_range, col_range, threshold=0, amplitude=1.0):
        """Creates a 2D gaussian filter (kernel) for the given dimensions which can be used

        :param row_range: array of times
        :param col_range: array of frequency centers
        :param threshold:
        :param amplitude:
        :return: A Kernel2D object
        """
        (y, x) = np.meshgrid(row_range, col_range)
        #(y, x) = np.meshgrid(np.arange(self.translate-4*self.sigma_f, self.translate+4*self.sigma_f,step_y),
        # np.arange(-4*self.sigma_t, 4*self.sigma_t,step_x), indexing='ij')
        # eventually use sigma_x and sigma_y

        #gauss = np.exp(-.5 * ((x * np.cos(self.theta) + (y-self.translate) * np.sin(self.theta)) ** 2 / self.sigma1 ** 2
        #                      + (-x * np.sin(self.theta) + (y-self.translate) * np.cos(self.theta)) ** 2 / self.sigma2 ** 2))
        translate_t = 0.5*self.Lambda*np.cos(self.theta)
        gauss = np.exp(-.5 * ((x-translate_t)**2/self.sigma_t**2 + (y-self.translate)**2/self.sigma_f**2))
        wave = np.cos(2 * np.pi / self.Lambda * ((x-translate_t) * np.cos(self.theta) + (y-self.translate) * np.sin(self.theta)) + self.psi)
        gb = gauss * wave
        gb *= amplitude / np.max(gb)
        print('max: ', np.max(gb))
        print('min:', np.min(gb))

        # Need to translate

        threshold = 0.05 * np.max(gb)
        kernel = Kernel2D.from_dense(row_range, col_range, gb, threshold=threshold)
        #kernel.apply_threshold(threshold)      # Already applied?
        print('min after thresh: ', np.min(gb))
        kernel.normalize2()
        kernel.kernel *= amplitude      # How do normalize and amplitude work together? seems like they would counteract each other?
        #kernel.imshow()

        return kernel

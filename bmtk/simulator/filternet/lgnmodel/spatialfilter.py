import ast
import numpy as np
import itertools
from six import string_types
from scipy import ndimage

from . import utilities as util
from .kernel import Kernel2D


class ArrayFilter(object):
    def __init__(self, mask):
        self.mask = mask
        
    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range, threshold).imshow(**kwargs)
        
    def get_kernel(self, row_range, col_range, threshold=0, amplitude=1.):
        row_vals, col_vals = np.where(self.mask > threshold)

        kernel_vals = self.mask[row_vals, col_vals]
        kernel_vals = amplitude*kernel_vals/kernel_vals.sum()
        
        return Kernel2D(row_range, col_range, row_vals, col_vals, kernel_vals)
    

class GaussianSpatialFilter(object):
    def __init__(self, translate=(0.0, 0.0), sigma=(1.0, 1.0), rotation=0, origin='center'):
        """A 2D gaussian used for filtering a part of the receptive field.

        :param translate: (float, float), the location of the gaussian on the screen relative to the origin in pixels.
        :param sigma: (float, float), the x and y gaussian std
        :param rotation: rotation of the gaussian in degrees
        :param origin: origin of the receptive field (defaults center of image)
        """

        # When w=1 and rotation=0, half-height will be at y=1
        self.translate = translate
        self.rotation = rotation

        if isinstance(sigma, string_types):
            # TODO: Move this to calling method
            try:
                sigma = ast.literal_eval(sigma)
            except Exception as exc:
                pass

        self.sigma = sigma
        self.origin = origin
        
    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range, threshold).imshow(**kwargs)
        
    def to_dict(self):
        return {
            'class': (__name__, self.__class__.__name__),
            'translate': self.translate,
            'rotation': self.rotation,
            'sigma': self.sigma
        }
        
    def get_kernel(self, row_range, col_range, threshold=0, amplitude=1.0):
        """Creates a 2D gaussian filter (kernel) for the given dimensions which can be used

        :param row_range: field height in pixels
        :param col_range: field width in pixels
        :param threshold:
        :param amplitude:
        :return: A Kernel2D object
        """

        # Create symmetric initial point at center:
        image_shape = len(col_range), len(row_range)
        h, w = image_shape
        on_filter_spatial = np.zeros(image_shape)
        if h % 2 == 0 and w % 2 == 0:
            for ii, jj in itertools.product(range(2), range(2)):
                on_filter_spatial[int(h/2)+ii-1, int(w/2)+jj-1] = .25
        elif h % 2 == 0 and w % 2 != 0:
            for ii in range(2):
                on_filter_spatial[int(h/2)+ii-1, int(w/2)] = .25
        elif h % 2 != 0 and w % 2 == 0:
            for jj in range(2):
                on_filter_spatial[int(h/2), int(w/2)+jj-1] = .25
        else:
            on_filter_spatial[int(h/2), int(w/2)] = .25
        
        # Apply gaussian filter to create correct sigma:
        scaled_sigma_x = float(self.sigma[0]) / (col_range[1]-col_range[0])
        scaled_sigma_y = float(self.sigma[1]) / (row_range[1]-row_range[0])
        on_filter_spatial = ndimage.gaussian_filter(on_filter_spatial, (scaled_sigma_x, scaled_sigma_y), mode='nearest',
                                                    cval=0)

        # Rotate and translate gaussian at center:
        rotation_matrix = util.get_rotation_matrix(self.rotation, on_filter_spatial.shape)
        translation_x = float(self.translate[1])/(row_range[1] - row_range[0])
        translation_y = -float(self.translate[0])/(col_range[1] - col_range[0])
        translation_matrix = util.get_translation_matrix((translation_x, translation_y))
        if self.origin != 'center':
            center_y = -(self.origin[0] - (col_range[-1] + col_range[0])/2)/(col_range[1] - col_range[0])
            center_x = (self.origin[1] - (row_range[-1] + row_range[0])/2)/(row_range[1] - row_range[0])
            translation_matrix += util.get_translation_matrix((center_x, center_y))
        kernel_data = util.apply_transformation_matrix(on_filter_spatial, translation_matrix + rotation_matrix)
        
        kernel = Kernel2D.from_dense(row_range, col_range, kernel_data, threshold=0)
        kernel.apply_threshold(threshold)
        kernel.normalize()
        kernel.kernel *= amplitude

        return kernel

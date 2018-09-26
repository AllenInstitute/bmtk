from scipy import ndimage
import numpy as np
import itertools
import importlib
import scipy.interpolate as spinterp
from . import utilities as util
import matplotlib.pyplot as plt
import scipy.misc as spmisc
import scipy.ndimage as spndimage
from .kernel import Kernel2D, Kernel3D

class ArrayFilter(object):
    
    default_threshold = .01
    
    def __init__(self, mask):
        
        self.mask = mask
        
    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        
        return self.get_kernel(row_range, col_range,threshold).imshow(**kwargs)
        
    def get_kernel(self, row_range, col_range, threshold=0, amplitude=1.):
        
#         print np.where(self.mask>threshold)
        row_vals, col_vals = np.where(self.mask>threshold)

        kernel_vals = self.mask[row_vals, col_vals]
        kernel_vals = amplitude*kernel_vals/kernel_vals.sum()
        
        return Kernel2D(row_range, col_range, row_vals, col_vals, kernel_vals)    # row_range, col_range, row_inds, col_inds, kernel):
    

class GaussianSpatialFilter(object):
    
    default_threshold = .01
     
    def __init__(self, translate=(0, 0), sigma=(1.,1.), rotation=0, origin='center'):
        '''When w=1 and rotation=0, half-height will be at y=1'''

        self.translate = translate
        self.rotation = rotation
        self.sigma = sigma 
        self.origin = origin
        
    def imshow(self, row_range, col_range, threshold=0, **kwargs):
        return self.get_kernel(row_range, col_range,threshold).imshow(**kwargs)
        
    def to_dict(self):

        return {'class':(__name__, self.__class__.__name__),
                'translate':self.translate,
                'rotation':self.rotation,
                'sigma':self.sigma}
        
    def get_kernel(self, row_range, col_range, threshold=0, amplitude=1.):

        # Create symmetric initial point at center:
        image_shape = len(col_range), len(row_range)
        h, w = image_shape
        on_filter_spatial = np.zeros(image_shape)
        if h%2 == 0 and w%2 == 0:
            for ii, jj in itertools.product(range(2), range(2)):
                on_filter_spatial[int(h/2)+ii-1,int(w/2)+jj-1] = .25
        elif h%2 == 0 and w%2 != 0:
            for ii in range(2):
                on_filter_spatial[int(h/2)+ii-1,int(w/2)] = .25
        elif h%2 != 0 and w%2 == 0:
            for jj in range(2):
                on_filter_spatial[int(h/2),int(w/2)+jj-1] = .25
        else:
            on_filter_spatial[int(h/2),int(w/2)] = .25
        
        # Apply gaussian filter to create correct sigma:
        scaled_sigma_x =float(self.sigma[0])/(col_range[1]-col_range[0])
        scaled_sigma_y = float(self.sigma[1])/(row_range[1]-row_range[0]) 
        on_filter_spatial = ndimage.gaussian_filter(on_filter_spatial, (scaled_sigma_x, scaled_sigma_y), mode='nearest', cval=0)
#         on_filter_spatial = skf.gaussian_filter(on_filter_spatial, sigma=(scaled_sigma_x, scaled_sigma_y))
        
        # Rotate and translate at center:
        rotation_matrix = util.get_rotation_matrix(self.rotation, on_filter_spatial.shape)
        translation_x = float(self.translate[1])/(row_range[1]-row_range[0])
        translation_y = -float(self.translate[0])/(col_range[1]-col_range[0])
        translation_matrix = util.get_translation_matrix((translation_x, translation_y))
        if self.origin != 'center':
            center_y = -(self.origin[0]-(col_range[-1]+col_range[0])/2)/(col_range[1]-col_range[0])
            center_x = (self.origin[1]-(row_range[-1]+row_range[0])/2)/(row_range[1]-row_range[0])
            translation_matrix += util.get_translation_matrix((center_x, center_y))
        kernel_data = util.apply_transformation_matrix(on_filter_spatial, translation_matrix+rotation_matrix)
        
        kernel = Kernel2D.from_dense(row_range, col_range, kernel_data, threshold=0)
        kernel.apply_threshold(threshold)
        kernel.normalize()
       
        kernel.kernel *= amplitude
        
        
        return kernel
        


# spatial_model = GaussianSpatialFilterModel(height=21, aspect_ratio=1., rotation=0)
# spatial_filter = spatial_model(center=(30,40))
# k = spatial_filter.get_spatial_kernel(range(60), range(80))
# k.imshow(frame_size=(60,80))













#     def evaluate_movie(self, movie, t, show=False):
#         
#         y = []
#         for ti in t:
#             kernel_result = movie.evaluate_Kernel3D(ti, self)
#             y.append(self.transfer_function(kernel_result))
#             
#         if show == True:
#             plt.plot(t, y)
#             plt.show()
#         
#         return t, y
        
#         print mesh_range[0]
#         
#         ii = mesh_range[0][inds]
#         jj = mesh_range[1][inds]
#         print ii, jj
#         print tmp[jj,ii]

#         plt.figure()
#         plt.pcolor(mesh_range[0], mesh_range[1], tmp)
#         plt.colorbar()
#         plt.axis('equal')
#         plt.show()

#         print self.xydata[0].shape
# 
#         t0 = spndimage.rotate(self.xydata[0],30,reshape=False, mode=mode)
#         t1 = spndimage.rotate(self.xydata[1],30, reshape=False, mode=mode)
        
#         print t0.shape
#         print t1.shape
#         print on_filter_spatial.shape
        
#         plt.pcolor(t0,t1, on_filter_spatial)
        
        
#         self.interpolation_function = spinterp.interp2d(self.w_values, self.h_values, on_filter_spatial, fill_value=0, bounds_error=False)
# 
#         print self.interpolation_function((t0,t1))

#         translation_matrix = util.get_translation_matrix(self.translation)
#         tmp = util.apply_transformation_matrix(on_filter_spatial, translation_matrix)
#         
#         plt.pcolor(self.xydata[0], self.xydata[1], tmp)
#         plt.show()
        
# #         print self.xydata_trans[0][0], self.xydata_trans[0],[-1]
# #         print self.xydata_trans[1][0], self.xydata_trans[1],[-1]
#         print self.xydata_trans
#         rotation_matrix = util.get_rotation_matrix(self.rotation, on_filter_spatial.shape)
#         translation_matrix = util.get_translation_matrix(self.translation)
#         on_filter_spatial = util.apply_transformation_matrix(on_filter_spatial, translation_matrix+rotation_matrix)

#         plt.imshow(on_filter_spatial, extent=(self.w_values[0], self.w_values[-1], self.h_values[0], self.h_values[-1]), aspect=1.)
#         plt.show()
        
#     def to_dict(self):
#         
#         return {'scale':self.scale,
#                 'translation':self.translation,
#                 'rotation':self.rotation,
#                 'weight':self.weight,
#                 'temporal_filter':self.temporal_filter.to_dict(),
#                 'class':(__name__, self.__class__.__name__)}

#     def get_kernel(self, xdata, ydata, threshold=default_threshold):
# 
# 
#         # Rotate and translate at center:
#         rotation_matrix = util.get_rotation_matrix(self.rotation, on_filter_spatial.shape)
#         translation_matrix = util.get_translation_matrix(self.translation)
#         on_filter_spatial = util.apply_transformation_matrix(on_filter_spatial, translation_matrix+rotation_matrix)
#         
#         # Now translate center of field in image:
# #         translation_matrix = util.get_translation_matrix(relative_spatial_location)
# #         on_filter_spatial = util.apply_transformation_matrix(on_filter_spatial, translation_matrix)
# 
#         # Create and return thresholded 2D mask:
#         row_ind_list, col_ind_list = np.where(on_filter_spatial != 0)
#         kernel = on_filter_spatial[row_ind_list, col_ind_list]
#         
#         
#         
#         
# #         filter_mask = Kernel2D(row_ind_list, col_ind_list, kernel, threshold=threshold)        
# 
#         return filter_mask

#         translation_matrix = util.get_translation_matrix((1.*translation[0]/fudge_factor,-1.*translation[1]/fudge_factor))

#         plt.figure()
#         plt.pcolor(self.mesh_support[0], self.mesh_support[1], self.kernel_data)
#         plt.axis('equal')
#         plt.show()
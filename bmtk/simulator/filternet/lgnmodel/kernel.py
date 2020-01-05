import matplotlib as mpl
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


def find_l_r_in_t_range(t_range, t):
    for tl in range(len(t_range)-1):
        tr = tl+1        
        test_val = (t_range[tl]-t)*(t_range[tr]-t)
        if np.abs(test_val) < 1e-16:
            
            if np.abs(t_range[tl]-t) < 1e-16:
                return (tl,)
            else:
                return (tr,)
            
        elif test_val < 0:
            t_range[tl], t_range[tr], t
            return tl, tr    


def get_contour(X, Y, Z, c):
    contour_obj = plt.contour(X, Y, Z)
    res = contour_obj.trace(c)
    nseg = len(res) // 2
    if nseg > 0:
        seg = res[:nseg][0]
        return seg[:, 0], seg[:, 1]
    else:
        return [], []


def plot_single_contour(ax, x_contour, y_contour, t, color):
    t_contour = t+np.zeros_like(x_contour)
    ax.plot(x_contour, t_contour, y_contour, zdir='z', color=color)
    

class Kernel1D(object):
    def __init__(self, t_range, kernel_array, threshold=0., reverse=False):
        assert len(t_range) == len(kernel_array)

        kernel_array = np.array(kernel_array)
        inds_to_keep = np.where(np.abs(kernel_array) > threshold)

        if reverse:
            self.t_range = -np.array(t_range)[::-1]

            t_inds_tmp = inds_to_keep[0]
            max_t_ind = t_inds_tmp.max()
            reversed_t_inds = max_t_ind - t_inds_tmp
            self.t_inds = reversed_t_inds - max_t_ind - 1  # Had an off by one error here should be "- 1" nhc 14 Apr '17 change made in cursor evalutiate too

        else:
            self.t_range = np.array(t_range)
            self.t_inds = inds_to_keep[0]

        self.kernel = kernel_array[inds_to_keep]
        assert len(self.t_inds) == len(self.kernel)

    def rescale(self):
        if np.abs(self.kernel.sum())!=0:
            self.kernel /= np.abs(self.kernel.sum())
    
    def normalize(self):
        self.kernel /= np.abs(self.kernel.sum())
    
    def __len__(self):
        return len(self.kernel)
        
    def imshow(self, ax=None, show=True, save_file_name=None, ylim=None, xlim=None, color='b', reverse=True):
        
        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        t_vals = self.t_range[self.t_inds]
        kernel_data = self.kernel
        if reverse:
            kernel_data = self.kernel[-1::-1]

        ax.plot(t_vals, kernel_data, color)
        ax.set_xlabel('Time (Seconds)')
        
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            a, b = (t_vals[0], t_vals[-1])
            ax.set_xlim(min(a, b), max(a, b))
        
        if save_file_name is not None:
            ax.savefig(save_file_name, transparent=True)

        if show:
            plt.show()
        
        return ax, (t_vals, self.kernel)

    def full(self, truncate_t=True):
        data = np.zeros(len(self.t_range))
        data[self.t_inds] = self.kernel

        if truncate_t:
            ind_min = np.where(np.abs(data) > 0)[0].min()
            return data[ind_min:]
        else:
            return data

        return data


class Kernel2D(object):
    def __init__(self, row_range, col_range, row_inds, col_inds, kernel):

        self.col_range = np.array(col_range)
        self.row_range = np.array(row_range)
        self.row_inds = np.array(row_inds)
        self.col_inds = np.array(col_inds)

        self.kernel = np.array(kernel)

        assert len(self.row_inds) == len(self.col_inds)
        assert len(self.row_inds) == len(self.kernel)

    def rescale(self):
        if np.abs(self.kernel.sum()) != 0:
            self.kernel /= np.abs(self.kernel.sum())
    
    def normalize(self):
        self.kernel /= np.abs(self.kernel.sum())
            
    @classmethod
    def from_dense(cls, row_range, col_range, kernel_array, threshold=0.):
        col_range = np.array(col_range).copy()
        row_range = np.array(row_range).copy()
        kernel_array = np.array(kernel_array).copy()
        inds_to_keep = np.where(np.abs(kernel_array) > threshold)
        kernel = kernel_array[inds_to_keep]
        if len(inds_to_keep) == 1:
            col_inds, row_inds = np.array([]), np.array([])
        else:
            col_inds, row_inds = inds_to_keep
        
        return cls(row_range, col_range, row_inds, col_inds,  kernel)
    
    @classmethod
    def copy(cls, instance):
        return cls(instance.row_range.copy(), 
                   instance.col_range.copy(), 
                   instance.row_inds.copy(),
                   instance.col_inds.copy(),
                   instance.kernel.copy())

    def __mul__(self, constant):
        new_copy = Kernel2D.copy(self)
        new_copy.kernel *= constant
        return new_copy
    
    def __add__(self, other):
        if len(other) == 0:
            return self

        try:
            np.testing.assert_almost_equal(self.row_range, other.row_range)
            np.testing.assert_almost_equal(self.col_range, other.col_range)
        except:
            raise Exception('Kernels must exist on same grid to be added')
        
        row_range = self.row_range.copy()
        col_range = self.col_range.copy()
        
        kernel_dict = {}
        for key, ker in zip(zip(self.row_inds, self.col_inds), self.kernel):
            kernel_dict[key] = kernel_dict.setdefault(key, 0) + ker 
        for key, ker in zip(zip(other.row_inds, other.col_inds), other.kernel):
            kernel_dict[key] = kernel_dict.setdefault(key, 0) + ker 
        
        key_list, kernel_list = zip(*kernel_dict.items())
        row_inds_list, col_inds_list = zip(*key_list)
        row_inds = np.array(row_inds_list)
        col_inds = np.array(col_inds_list)
        kernel = np.array(kernel_list)
        
        return Kernel2D(row_range, col_range, row_inds, col_inds, kernel)
    
    def apply_threshold(self, threshold):
        
        inds_to_keep = np.where(np.abs(self.kernel) > threshold)
        self.row_inds = self.row_inds[inds_to_keep]
        self.col_inds = self.col_inds[inds_to_keep]
        self.kernel = self.kernel[inds_to_keep]
        
    def full(self):
        data = np.zeros((len(self.row_range), len(self.col_range)))
        data[self.row_inds, self.col_inds] = self.kernel
        return data
        
    def imshow(self, ax=None, show=True, save_file_name=None, clim=None, colorbar=True):
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
        
        data = self.full()

        if clim is not None:
            im = ax.imshow(data, extent=(self.col_range[0], self.col_range[-1], self.row_range[0], self.row_range[-1]),
                           origin='lower', clim=clim, interpolation='none')
        else:
            im = ax.imshow(data, extent=(self.col_range[0], self.col_range[-1], self.row_range[0], self.row_range[-1]),
                           origin='lower', interpolation='none')
        
        if colorbar:
            plt.colorbar(im, cax=cax)

        if save_file_name is not None:
            plt.savefig(save_file_name, transparent=True)

        if show:
            plt.show()
            
        return ax, data
        
    def __len__(self):
        return len(self.kernel)


class Kernel3D(object):
    def rescale(self):
        if np.abs(self.kernel.sum()) != 0:
            self.kernel /= np.abs(self.kernel.sum())
    
    def normalize(self):
        self.kernel /= (self.kernel.sum())*np.sign(self.kernel.sum())

    @classmethod
    def copy(cls, instance):
        return cls(instance.row_range.copy(),
                   instance.col_range.copy(), 
                   instance.t_range.copy(),
                   instance.row_inds.copy(),
                   instance.col_inds.copy(),
                   instance.t_inds.copy(),  
                   instance.kernel.copy())
        
    def __len__(self):
        return len(self.kernel)
    
    def __init__(self, row_range, col_range, t_range, row_inds, col_inds, t_inds, kernel):
        
        self.col_range = np.array(col_range)
        self.row_range = np.array(row_range)
        self.t_range = np.array(t_range)
        self.col_inds = np.array(col_inds)
        self.row_inds = np.array(row_inds)
        self.t_inds = np.array(t_inds)
        self.kernel = np.array(kernel)
        
        assert len(self.row_inds) == len(self.col_inds)
        assert len(self.row_inds) == len(self.t_inds)
        assert len(self.row_inds) == len(self.kernel)
        
    def apply_threshold(self, threshold):
        
        inds_to_keep = np.where(np.abs(self.kernel) > threshold)
        self.row_inds = self.row_inds[inds_to_keep]
        self.col_inds = self.col_inds[inds_to_keep]
        self.t_inds = self.t_inds[inds_to_keep]
        self.kernel = self.kernel[inds_to_keep]
        
    def __add__(self, other):
        if len(other) == 0:
            return self

        try:
            if not (len(self.row_range) == 0 or len(other.row_range) == 0): 
                np.testing.assert_almost_equal(self.row_range, other.row_range)
            if not (len(self.col_range) == 0 or len(other.col_range) == 0):
                np.testing.assert_almost_equal(self.col_range, other.col_range)
            if not (len(self.t_range) == 0 or len(other.t_range) == 0):
                np.testing.assert_almost_equal(self.t_range, other.t_range)
        except:
            raise Exception('Kernels must exist on same grid to be added')
        
        if len(self.row_range) == 0:
            row_range = other.row_range.copy()
        else:
            row_range = self.row_range.copy()
        if len(self.col_range) == 0:
            col_range = other.col_range.copy()
        else:
            col_range = self.col_range.copy()
        if len(self.t_range) == 0:
            t_range = other.t_range.copy()
        else:
            t_range = self.t_range.copy()
        
        kernel_dict = {}
        for key, ker in zip(zip(self.row_inds, self.col_inds, self.t_inds), self.kernel):
            kernel_dict[key] = kernel_dict.setdefault(key, 0) + ker 
        for key, ker in zip(zip(other.row_inds, other.col_inds, other.t_inds), other.kernel):
            kernel_dict[key] = kernel_dict.setdefault(key, 0) + ker 
        
        key_list, kernel_list = zip(*kernel_dict.items())
        row_inds_list, col_inds_list, t_inds_list = zip(*key_list)
        row_inds = np.array(row_inds_list)
        col_inds = np.array(col_inds_list)
        t_inds = np.array(t_inds_list)
        kernel = np.array(kernel_list)
        
        return Kernel3D(row_range, col_range, t_range, row_inds, col_inds, t_inds, kernel)
    
    def __mul__(self, constant):
        
        new_copy = Kernel3D.copy(self)
        new_copy.kernel *= constant
        return new_copy
    
    def t_slice(self, t):
        
        ind_list = find_l_r_in_t_range(self.t_range, t)

        if ind_list is None:
            return None

        elif len(ind_list) == 1:
            
            t_ind_i = ind_list[0]
            inds_i = np.where(self.t_range[self.t_inds] == self.t_range[t_ind_i])
            row_inds = self.row_inds[inds_i]
            col_inds = self.col_inds[inds_i]
            kernel = self.kernel[inds_i]
            return Kernel2D(self.row_range, self.col_range, row_inds, col_inds, kernel)

        else:
            t_ind_l, t_ind_r = ind_list
            t_l, t_r = self.t_range[t_ind_l], self.t_range[t_ind_r]
            
            inds_l = np.where(self.t_range[self.t_inds] == self.t_range[t_ind_l])
            inds_r = np.where(self.t_range[self.t_inds] == self.t_range[t_ind_r])
            row_inds_l = self.row_inds[inds_l]
            col_inds_l = self.col_inds[inds_l]
            kernel_l = self.kernel[inds_l]
            kl = Kernel2D(self.row_range, self.col_range, row_inds_l, col_inds_l, kernel_l)
            row_inds_r = self.row_inds[inds_r]
            col_inds_r = self.col_inds[inds_r]
            kernel_r = self.kernel[inds_r]
            kr = Kernel2D(self.row_range, self.col_range, row_inds_r,  col_inds_r, kernel_r)
            wa, wb = (1-(t-t_l)/(t_r-t_l)), (1-(t_r-t)/(t_r-t_l))
            
            return kl*wa + kr*wb

    def full(self, truncate_t=True):

        data = np.zeros((len(self.t_range), len(self.row_range), len(self.col_range)))
        data[self.t_inds, self.row_inds, self.col_inds] = self.kernel

        if truncate_t:
            ind_max = np.where(np.abs(data) > 0)[0].min()
            return data[ind_max:, :, :]
        else:
            return data

    def imshow(self, ax=None, t_range=None, cmap=cm.bwr, N=10, show=True, save_file_name=None, kvals=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        
        if t_range is None:
            t_range = self.t_range
        
        slice_list_sparse = [self.t_slice(t) for t in t_range]
        slice_list = []
        slice_t_list = []
        for curr_slice, curr_t in zip(slice_list_sparse, t_range):
            if not curr_slice is None:
                slice_list.append(curr_slice.full())
                slice_t_list.append(curr_t)
        all_slice_max = max(map(np.max, slice_list))
        all_slice_min = min(map(np.min, slice_list))
        upper_bound = max(np.abs(all_slice_max), np.abs(all_slice_min))
        lower_bound = -upper_bound 
        norm = mpl.colors.Normalize(vmin=lower_bound, vmax=upper_bound)
        color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba
        
        if kvals is None:
            kvals = np.linspace(lower_bound, upper_bound, N)
        
        X, Y = np.meshgrid(self.row_range, self.col_range)

        contour_dict = {}
        for kval in kvals:
            for t_val, curr_slice in zip(slice_t_list, slice_list):
                x_contour, y_contour = get_contour(Y, X, curr_slice.T, kval)
                contour_dict[kval, t_val] = x_contour, y_contour 
                color = color_mapper(kval)
                color = color[0], color[1], color[2], np.abs(kval)/upper_bound
                plot_single_contour(ax, x_contour, y_contour, t_val, color)
        
        ax.set_zlim(self.row_range[0], self.row_range[-1])
        ax.set_ylim(self.t_range[0], self.t_range[-1])
        ax.set_xlim(self.col_range[0], self.col_range[-1])
        
        if save_file_name is not None:
            plt.savefig(save_file_name, transparent=True)
        
        if show:
            plt.show() 
        
        return ax, contour_dict


def merge_spatial_temporal(spatial_kernel, temporal_kernel, threshold=0):
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
        
        return kernel

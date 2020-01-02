from .kernel import Kernel2D, Kernel3D
from .cursor import LNUnitCursor, MultiLNUnitCursor, MultiLNUnitMultiMovieCursor, SeparableLNUnitCursor, \
    SeparableMultiLNUnitCursor

    
class LNUnit(object):
    def __init__(self, linear_filter, transfer_function, amplitude=1.):
        self.linear_filter = linear_filter
        self.transfer_function = transfer_function
        self.amplitude = amplitude

    def evaluate(self, movie, **kwargs):
        return self.get_cursor(movie, separable=kwargs.pop('separable', False)).evaluate(**kwargs)
 
    def get_spatiotemporal_kernel(self, *args, **kwargs):
        return self.linear_filter.get_spatiotemporal_kernel(*args, **kwargs)
    
    def get_cursor(self, movie, threshold=0, separable=False):
        if separable:
            return SeparableLNUnitCursor(self, movie)
        else:
            return LNUnitCursor(self, movie, threshold=threshold)
    
    def show_temporal_filter(self, *args, **kwargs):
        self.linear_filter.show_temporal_filter(*args, **kwargs)
        
    def show_spatial_filter(self, *args, **kwargs):
        self.linear_filter.show_spatial_filter(*args, **kwargs)
    
    def to_dict(self):
        return {
            'class': (__name__, self.__class__.__name__),
            'linear_filter': self.linear_filter.to_dict(),
            'transfer_function': self.transfer_function.to_dict()
        }


class MultiLNUnit(object):
    def __init__(self, lnunit_list, transfer_function):
        self.lnunit_list = lnunit_list
        self.transfer_function = transfer_function
        
    def get_spatiotemporal_kernel(self, *args, **kwargs):
        k = Kernel3D([], [], [], [], [], [], [])
        for unit in self.lnunit_list:
            k = k+unit.get_spatiotemporal_kernel(*args, **kwargs)
            
        return k
        
    def show_temporal_filter(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        ax = kwargs.pop('ax', None)
        show = kwargs.pop('show', None)
        save_file_name = kwargs.pop('save_file_name', None) 

        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        kwargs.update({'ax': ax, 'show': False, 'save_file_name': None})
        for unit in self.lnunit_list:
            if unit.linear_filter.amplitude < 0:
                color = 'b'
            else:
                color = 'r'
            unit.linear_filter.show_temporal_filter(color=color, **kwargs)

        if save_file_name is not None:
            plt.savefig(save_file_name, transparent=True)
         
        if show:
            plt.show()
         
        return ax
    
    def show_spatial_filter(self, *args, **kwargs):
        
        ax = kwargs.pop('ax', None)
        show = kwargs.pop('show', True)
        save_file_name = kwargs.pop('save_file_name', None) 
        colorbar = kwargs.pop('colorbar', True) 
        
        k = Kernel2D(args[0], args[1], [], [], [])
        for lnunit in self.lnunit_list:
            k = k + lnunit.linear_filter.spatial_filter.get_kernel(*args, **kwargs)
        k.imshow(ax=ax, show=show, save_file_name=save_file_name, colorbar=colorbar)
        
    def get_cursor(self, *args, **kwargs):
        
        threshold = kwargs.get('threshold', 0.)
        separable = kwargs.get('separable', False)
        
        if len(args) == 1:
            movie = args[0]
            if separable:
                return SeparableMultiLNUnitCursor(self, movie)
            else:
                return MultiLNUnitCursor(self, movie, threshold=threshold)
        elif len(args) > 1:
            movie_list = args
            if separable:
                raise NotImplementedError
            else:
                return MultiLNUnitMultiMovieCursor(self, movie_list, threshold=threshold)
        else:
            assert ValueError
    
    def evaluate(self, movie, **kwargs):
        seperable = kwargs.pop('separable', False)
        return self.get_cursor(movie, separable=seperable).evaluate(**kwargs)

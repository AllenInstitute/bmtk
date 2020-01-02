import os
import numpy as np
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

from .linearfilter import SpatioTemporalFilter
from .spatialfilter import GaussianSpatialFilter
from .temporalfilter import TemporalFilterCosineBump
from .movie import Movie
from .lgnmodel1 import LGNModel, heat_plot
from .transferfunction import MultiTransferFunction, ScalarTransferFunction
from .lnunit import LNUnit, MultiLNUnit


class OnUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude > 0
        super(OnUnit, self).__init__(linear_filter, transfer_function)


class OffUnit(LNUnit):
    def __init__(self, linear_filter, transfer_function):
        assert linear_filter.amplitude < 0
        super(OffUnit, self).__init__(linear_filter, transfer_function)


class LGNOnOffCell(MultiLNUnit):
    """A cell model for a OnOff cell"""
    def __init__(self, on_filter, off_filter,
                 transfer_function=MultiTransferFunction((symbolic_x, symbolic_y),
                                                         'Heaviside(x)*(x)+Heaviside(y)*(y)')):
        """Summary

        :param on_filter:
        :param off_filter:
        :param transfer_function:
        """
        self.on_filter = on_filter
        self.off_filter = off_filter
        self.on_unit = OnUnit(self.on_filter, ScalarTransferFunction('s'))
        self.off_unit = OffUnit(self.off_filter, ScalarTransferFunction('s'))
        super(LGNOnOffCell, self).__init__([self.on_unit, self.off_unit], transfer_function)
        

class TwoSubfieldLinearCell(MultiLNUnit):
    def __init__(self, dominant_filter, nondominant_filter, subfield_separation=10, onoff_axis_angle=45,
                 dominant_subfield_location=(30, 40),
                 transfer_function=MultiTransferFunction((symbolic_x, symbolic_y),
                                                         'Heaviside(x)*(x)+Heaviside(y)*(y)')):
        self.subfield_separation = subfield_separation
        self.onoff_axis_angle = onoff_axis_angle
        self.dominant_subfield_location = dominant_subfield_location
        self.dominant_filter = dominant_filter
        self.nondominant_filter = nondominant_filter
        self.transfer_function = transfer_function

        self.dominant_unit = LNUnit(self.dominant_filter, ScalarTransferFunction('s'),
                                    amplitude=self.dominant_filter.amplitude)
        self.nondominant_unit = LNUnit(self.nondominant_filter, ScalarTransferFunction('s'),
                                       amplitude=self.dominant_filter.amplitude)

        super(TwoSubfieldLinearCell, self).__init__([self.dominant_unit, self.nondominant_unit],
                                                    self.transfer_function)
              
        self.dominant_filter.spatial_filter.translate = self.dominant_subfield_location
        hor_offset = np.cos(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation + self.dominant_subfield_location[0]
        vert_offset = np.sin(self.onoff_axis_angle*np.pi/180.)*self.subfield_separation + self.dominant_subfield_location[1]
        rel_translation = (hor_offset, vert_offset)
        self.nondominant_filter.spatial_filter.translate = rel_translation


"""
class LGNOnCell(OnUnit):
    def __init__(self, **kwargs):
        self.position = kwargs.pop('position', None)
        self.weights = kwargs.pop('weights', None)
        self.kpeaks = kwargs.pop('kpeaks', None)
        self.delays = kwargs.pop('delays', None)
        self.amplitude = kwargs.pop('amplitude', None)
        self.sigma = kwargs.pop('sigma', None)
        self.transfer_function_str = kwargs.pop('transfer_function_str', 's')  # 'Heaviside(s)*s')
        self.metadata = kwargs.pop('metadata', {})

        temporal_filter = TemporalFilterCosineBump(self.weights, self.kpeaks, self.delays)
        spatial_filter = GaussianSpatialFilter(translate=self.position, sigma=self.sigma,
                                               origin=(0, 0))  # all distances measured from BOTTOM LEFT
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=self.amplitude)
        transfer_function = ScalarTransferFunction(self.transfer_function_str)
        self.unit = OnUnit(spatiotemporal_filter, transfer_function)


class LGNOffCell(OffUnit):
    def __init__(self, **kwargs):
        lattice_unit_center = kwargs.pop('lattice_unit_center', None)
        weights = kwargs.pop('weights', None)
        kpeaks = kwargs.pop('kpeaks', None)
        amplitude = kwargs.pop('amplitude', None)
        sigma = kwargs.pop('sigma', None)
        width = kwargs.pop('width', 5)
        transfer_function_str = kwargs.pop('transfer_function_str', 'Heaviside(s)*s')

        dxi = np.random.uniform(-width*1./2, width*1./2)
        dyi = np.random.uniform(-width*1./2, width*1./2)
        temporal_filter = TemporalFilterCosineBump(weights, kpeaks)
        spatial_filter = GaussianSpatialFilter(translate=(dxi, dyi), sigma=sigma, origin=lattice_unit_center)  # all distances measured from BOTTOM LEFT
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
        transfer_function = ScalarTransferFunction(transfer_function_str)
        super(LGNOnCell, self).__init__(spatiotemporal_filter, transfer_function)
"""

if __name__ == "__main__":
    movie_file = '/data/mat/iSee_temp_shared/movies/TouchOfEvil.npy'
    m_data = np.load(movie_file, 'r')
    m = Movie(m_data[1000:], frame_rate=30.)
    
    # Create second cell:
    transfer_function = ScalarTransferFunction('s')
    temporal_filter = TemporalFilterCosineBump((0.4, -0.3), (20, 60))
    cell_list = []
    for xi in np.linspace(0, m.data.shape[2], 5):
        for yi in np.linspace(0, m.data.shape[1], 5):
            spatial_filter_on = GaussianSpatialFilter(sigma=(2, 2), origin=(0, 0), translate=(xi, yi))
            on_linear_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter, amplitude=20)
            spatial_filter_off = GaussianSpatialFilter(sigma=(4, 4), origin=(0, 0), translate=(xi, yi))
            off_linear_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter, amplitude=-20) 
            on_off_cell = LGNOnOffCell(on_linear_filter, off_linear_filter)
            cell_list.append(on_off_cell)
    
    lgn = LGNModel(cell_list)  # Here include a list of all cells
    y = lgn.evaluate(m, downsample=100)  # Does the filtering + non-linearity on movie object m
    heat_plot(y, interpolation='none', colorbar=True)

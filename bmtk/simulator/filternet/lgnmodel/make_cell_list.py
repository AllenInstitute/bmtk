import scipy.stats as sps
import numpy as np

from .linearfilter import SpatioTemporalFilter
from .spatialfilter import GaussianSpatialFilter
from .temporalfilter import TemporalFilterCosineBump
from .cellmodel import LGNOnCell, LGNOffCell,LGNOnOffCell, TwoSubfieldLinearCell, OnUnit, OffUnit
from .transferfunction import MultiTransferFunction, ScalarTransferFunction


def multi_cell_random_generator(cell_creation_function=None, **kwargs):
    
    sew_param_dict = {}
    static_param_dict = {}
    range_key_dict = {}
    for key, val in kwargs.items():
        if isinstance(val, (sps.rv_continuous, sps.rv_discrete)) or type(val) == type(sps.multivariate_normal()):
            sew_param_dict[key] = val
        elif isinstance(val, np.ndarray): 
            range_key_dict[key] = val
        else:
            static_param_dict[key] = val
    
    number_of_cells = static_param_dict.pop('number_of_cells', 1)
    
    for key, val in range_key_dict.items():
        assert len(val) == number_of_cells
    
    cell_list = []
    loop_keys, loop_lists = zip(*sew_param_dict.items())
    value_instance_list = zip(*map(lambda x: x.rvs(size=number_of_cells), loop_lists))
    for ii, curr_value_instance in enumerate(value_instance_list):
        param_dict = dict(zip(loop_keys, curr_value_instance))
        param_dict.update(static_param_dict)
        param_dict['number_of_cells'] = 1
        for range_key in range_key_dict:
            param_dict[range_key] = range_key_dict[range_key][ii]
        
        if cell_creation_function is None:
            cell_list.append(param_dict)
        else:
            cell_list += cell_creation_function(**param_dict)
        
    return cell_list
    
    
def make_single_unit_cell_list(number_of_cells=None,
                               lattice_unit_center=None,
                               weights=None,
                               kpeaks=None,
                               delays=None,
                               amplitude=None,
                               sigma=None,
                               width=5,
                               transfer_function_str='Heaviside(s)*s'):

    cell_list = []
    for _ in range(number_of_cells):
        dxi = np.random.uniform(-width*1./2, width*1./2)
        dyi = np.random.uniform(-width*1./2, width*1./2)
        temporal_filter = TemporalFilterCosineBump(weights, kpeaks, delays)
        spatial_filter = GaussianSpatialFilter(translate=(dxi, dyi), sigma=sigma,
                                               origin=lattice_unit_center)  # all distances measured from BOTTOM LEFT
        spatiotemporal_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=amplitude)
        transfer_function = ScalarTransferFunction(transfer_function_str)
        if amplitude > 0.:
            cell = OnUnit(spatiotemporal_filter, transfer_function)
        elif amplitude < 0.:
            cell = OffUnit(spatiotemporal_filter, transfer_function)            
        else:
            raise Exception
        
        cell_list.append(cell)
        
    return cell_list


def make_on_off_cell_list(number_of_cells=None,
                          lattice_unit_center=None,
                          weights_on=None,
                          weights_off=None,
                          kpeaks_on=None,
                          kpeaks_off=None,
                          delays_on=None,
                          delays_off=None,
                          amplitude_on=None,
                          amplitude_off=None,
                          sigma_on=None,
                          sigma_off=None,
                          subfield_separation=None,
                          ang=None,
                          dominant_subunit=None,
                          width=5,
                          transfer_function_str='Heaviside(x)*x + Heaviside(y)*y'):

    cell_list = []
    for _ in range(number_of_cells):
        
        dxi = np.random.uniform(-width*1./2, width*1./2)
        dyi = np.random.uniform(-width*1./2, width*1./2)
        
        dominant_subfield_location = (lattice_unit_center[0]+dxi, lattice_unit_center[1]+dyi)

        if dominant_subunit == 'on':            
            on_translate = dominant_subfield_location  # (0,0)
            off_translate = dominant_subfield_location  # nondominant_subfield_translation
            
        elif dominant_subunit == 'off':
            
            off_translate = dominant_subfield_location  # (0,0)
            on_translate = dominant_subfield_location  # nondominant_subfield_translation
            
        else:
            raise Exception
        
        on_origin = off_origin = (0, 0)  # dominant_subfield_location

        temporal_filter_on = TemporalFilterCosineBump(weights_on, kpeaks_on, delays_on)
        spatial_filter_on = GaussianSpatialFilter(translate=on_translate, sigma=sigma_on,
                                                  origin=on_origin)  # all distances measured from BOTTOM LEFT
        on_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter_on, amplitude=amplitude_on)
        
        temporal_filter_off = TemporalFilterCosineBump(weights_off, kpeaks_off, delays_off)
        spatial_filter_off = GaussianSpatialFilter(translate=off_translate, sigma=sigma_off,
                                                   origin=off_origin)  # all distances measured from BOTTOM LEFT
        off_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter_off, amplitude=amplitude_off)

        cell = TwoSubfieldLinearCell(on_filter, off_filter, subfield_separation=subfield_separation,
                                     onoff_axis_angle=ang, dominant_subfield_location=dominant_subfield_location)
        cell_list.append(cell)
        
    return cell_list

    
def evaluate_cell_and_plot(input_cell, input_movie, ax, show=False):
    import matplotlib.pyplot as plt

    t, y = input_cell.evaluate(input_movie, downsample=10)
    ax.plot(t, y)
     
    if show:
        plt.show()

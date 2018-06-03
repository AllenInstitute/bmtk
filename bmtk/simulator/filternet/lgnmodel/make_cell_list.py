import scipy.io as sio
import os 
import matplotlib.pyplot as plt
import isee_engine.nwb as nwb
from linearfilter import SpatioTemporalFilter
import numpy as np 
from spatialfilter import GaussianSpatialFilter
from transferfunction import ScalarTransferFunction
from temporalfilter import TemporalFilterCosineBump
from cursor import LNUnitCursor, MultiLNUnitCursor
from movie import Movie    
from lgnmodel1 import LGNModel, heat_plot
from cellmodel import LGNOnCell, LGNOffCell,LGNOnOffCell,TwoSubfieldLinearCell, OnUnit, OffUnit
from transferfunction import MultiTransferFunction, ScalarTransferFunction
from lnunit import LNUnit, MultiLNUnit    
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y
from kernel import Kernel3D
from movie import Movie, FullFieldFlashMovie
import itertools
import scipy.stats as sps

# def multi_cell_tensor_generator(cell_creation_function, **kwargs):
#     
#     sew_param_dict = {}
#     static_param_dict = {}
#     for key, val in kwargs.items():
#         if isinstance(val, (list, np.ndarray)):
#             sew_param_dict[key]=val
#         else:
#             static_param_dict[key]=val
#     
#     cell_list = []
#     loop_keys, loop_lists = zip(*sew_param_dict.items())
#     for param_tuple in itertools.product(*loop_lists): 
#         param_dict = dict(zip(loop_keys, param_tuple))
#         print param_dict
#         param_dict.update(static_param_dict)
#         cell_list += cell_creation_function(**param_dict)
# 
#     return cell_list

def multi_cell_random_generator(cell_creation_function=None, **kwargs):
    
    sew_param_dict = {}
    static_param_dict = {}
    range_key_dict = {}
    for key, val in kwargs.items():
        if isinstance(val, (sps.rv_continuous, sps.rv_discrete)) or type(val) == type(sps.multivariate_normal()):
            sew_param_dict[key]=val
        elif isinstance(val, np.ndarray): 
            range_key_dict[key] = val
        else:
            static_param_dict[key]=val
    
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
                               transfer_function_str = 'Heaviside(s)*s'):

    cell_list = []
    for _ in range(number_of_cells):
        dxi = np.random.uniform(-width*1./2,width*1./2)
        dyi = np.random.uniform(-width*1./2,width*1./2)
        temporal_filter = TemporalFilterCosineBump(weights, kpeaks,delays)
        spatial_filter = GaussianSpatialFilter(translate=(dxi,dyi), sigma=sigma,  origin=lattice_unit_center) # all distances measured from BOTTOM LEFT
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
                          delays_on = None,
                          delays_off = None,
                          amplitude_on=None,
                          amplitude_off=None,
                          sigma_on=None,
                          sigma_off=None,
                          subfield_separation=None,
                          ang=None,
                          dominant_subunit=None,
                          width=5,
                          transfer_function_str = 'Heaviside(x)*x + Heaviside(y)*y'):

    cell_list = []
    for _ in range(number_of_cells):
        
        dxi = np.random.uniform(-width*1./2,width*1./2)
        dyi = np.random.uniform(-width*1./2,width*1./2)
        
        dominant_subfield_location = (lattice_unit_center[0]+dxi, lattice_unit_center[1]+dyi)
#         hor_offset = np.cos(ang*np.pi/180.)*subfield_separation
#         vert_offset = np.sin(ang*np.pi/180.)*subfield_separation
#         nondominant_subfield_translation = (hor_offset,vert_offset)
        
        if dominant_subunit == 'on':            
            on_translate = dominant_subfield_location#(0,0)
            off_translate =  dominant_subfield_location#nondominant_subfield_translation
            
        elif dominant_subunit == 'off':
            
            off_translate = dominant_subfield_location#(0,0)
            on_translate =  dominant_subfield_location#nondominant_subfield_translation
            
        else:
            raise Exception
        
        on_origin = off_origin = (0,0)#dominant_subfield_location

        temporal_filter_on = TemporalFilterCosineBump(weights_on, kpeaks_on,delays_on)
        spatial_filter_on = GaussianSpatialFilter(translate=on_translate,sigma=sigma_on, origin=on_origin) # all distances measured from BOTTOM LEFT
        on_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter_on, amplitude=amplitude_on)
        
        temporal_filter_off = TemporalFilterCosineBump(weights_off, kpeaks_off,delays_off)
        spatial_filter_off = GaussianSpatialFilter(translate=off_translate,sigma=sigma_off, origin=off_origin) # all distances measured from BOTTOM LEFT
        off_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter_off, amplitude=amplitude_off)

#        cell = LGNOnOffCell(on_filter, off_filter, transfer_function=MultiTransferFunction((symbolic_x, symbolic_y), transfer_function_str))
        cell = TwoSubfieldLinearCell(on_filter,off_filter,subfield_separation=subfield_separation, onoff_axis_angle=ang, dominant_subfield_location=dominant_subfield_location)
        cell_list.append(cell)
        
    return cell_list

# amplitude_list = amplitude_dist.rvs(size=5)
# kpeak_list = kpeak_dist.rvs(size=5)
# cell_config = {'number_of_cells':5,
#                 'lattice_unit_center':(40,30),
#                 'weights':(.4,-.2),
#                 'kpeaks':kpeak_list,
#                 'amplitude':amplitude_list,
#                 'sigma':(4,4),
#                 'width':5}
# multi_cell_tensor_generator(make_single_unit_cell_list, **cell_config)


# amplitude_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
# kpeak_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
# 
# single_unit_cell_config = {'number_of_cells':10,
#                 'lattice_unit_center':(40,30),
#                 'weights':(.4,-.2),
#                 'kpeaks':kpeak_dist,
#                 'amplitude':amplitude_dist,
#                 'sigma':(4,4),
#                 'width':5}
# 
# 
# amplitude_on_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
# amplitude_off_dist = sps.rv_discrete(values=([-10,-15], [.5,.5]))
# kpeak_on_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
# kpeak_off_dist =  sps.multivariate_normal(mean=[100., 160.], cov=[[5.0, 0], [0, 5]])
# #ang_dist = sps.rv_discrete(values=(np.arange(0,360,45), 1./8*np.ones((1,8))))
# ang_dist = np.arange(0,360,45)
# 
# two_unit_cell_config={'number_of_cells':8,
#                       'lattice_unit_center':(40,30),
#                       'weights_on':(.4,-.2),
#                       'weights_off':(.4,-.1),
#                       'kpeaks_on':kpeak_on_dist,
#                       'kpeaks_off':kpeak_off_dist,
#                       'amplitude_on':20.,
#                       'amplitude_off':-10.,
#                       'sigma_on':(4,4),
#                       'sigma_off':(4,4),
#                       'subfield_separation':2.,
#                       'ang':ang_dist,
#                       'dominant_subunit':'on',
#                       'width':5}

    
def evaluate_cell_and_plot(input_cell, input_movie, ax, show=False):
    t, y = input_cell.evaluate(input_movie,downsample = 10)
    ax.plot(t, y)
     
    if show == True:
        plt.show()
 
 
# if __name__ == "__main__":
#      
#     # Create stimulus 0:
#     frame_rate = 60
#     m1 = FullFieldFlashMovie(np.arange(60), np.arange(80), 1., 3., frame_rate=frame_rate).full(t_max=3)
#     m2 = FullFieldFlashMovie(np.arange(60), np.arange(80), 0, 2, frame_rate=frame_rate, max_intensity=-1).full(t_max=2)
#     m3 = FullFieldFlashMovie(np.arange(60), np.arange(80), 0, 2., frame_rate=frame_rate).full(t_max=2)
#     m4 = FullFieldFlashMovie(np.arange(60), np.arange(80), 0, 2, frame_rate=frame_rate, max_intensity=0).full(t_max=2)
#     m0 = m1+m2+m3+m4
#      
#     # Create stimulus 1:
#     movie_file = '/data/mat/RamIyer/for_Anton/grating_ori0_res2.mat'
#     m_file = sio.loadmat(movie_file)
#     m_data_raw = m_file['mov_fine'].T
#     m_data = np.reshape(m_data_raw,(3000,64,128))
#     m1 = Movie(m_data, frame_rate=1000.)
#      
#     #Create stimulus 2:
#     movie_file = '/data/mat/iSee_temp_shared/TouchOfEvil_norm.npy'
#     m_data = np.load(movie_file, 'r')
#     m = Movie(m_data[1000:], frame_rate=30.)
#      
#     movie_list = [m0, m1, m2]
#      
#     #====================================================
#          
#     #Create cell list
#      
#     cell_list = []
#      
#     #On cells
#     params_tON = (5, (40,30), (.4,-.2),(40,80),20.,(4,4))
#     tON_list = make_single_unit_cell_list(*params_tON)
#     cell_list.append(tON_list)
#  
#     params_sON = (5, (40,30), (.4,-.1),(100,160),20.,(4,4))
#     sON_list = make_single_unit_cell_list(*params_sON)
#     cell_list.append(sON_list)
#           
#     #Off cells
#     params_tOFF = (5, (40,30), (.4,-.2),(40,80),-20.,(4,4))
#     tOFF_list = make_single_unit_cell_list(*params_tOFF)
#     cell_list.append(tOFF_list)
#      
#     params_sOFF = (5, (40,30), (.4,-.1),(100,160),-20.,(4,4))
#     sOFF_list = make_single_unit_cell_list(*params_sOFF)
#     cell_list.append(sOFF_list)
#      
#     #ONOFF cells
#     params_onoff = (5, (40,30),(.4, -.2),(.4,-.2),(40, 80),(50,100),20.,-20.,(4,4),(4,4),2.,0,'on')
#     onoff_list = make_on_off_cell_list(*params_onoff)
#     cell_list.append(onoff_list)
#          
#     #Two subunit cells
#     params_twosub = (5, (40,30),(.4, -.2),(.4,-.1),(40, 80),(100,160),20.,-10.,(4,2),(3,4),10.,90,'on')
#     twosub_list = make_on_off_cell_list(*params_twosub)
#     cell_list.append(twosub_list)
#      
#     #=====================================================
#     #Evaluate and plot responses
#     nc = len(movie_list)
#     nr = len(cell_list)
#     fig, axes = plt.subplots(nr,nc+2) 
#      
#     for curr_row, curr_cell in zip(axes, cell_list):
#         curr_cell.show_spatial_filter(np.arange(60),np.arange(80), ax=curr_row[0], show=False, colorbar=False)
#         curr_cell.show_temporal_filter(ax=curr_row[1], show=False)
#      
#     for curr_row, curr_cell in zip(axes, cell_list):
#         for curr_ax, curr_movie in zip(curr_row[2:], movie_list):
#             evaluate_cell_and_plot(curr_cell, curr_movie, curr_ax, show=False)
#  
# plt.tight_layout()                
# plt.show()

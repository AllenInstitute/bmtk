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
from lgnmodel import LGNModel, heat_plot
from cellmodel import LGNOnCell, LGNOffCell,LGNOnOffCell,TwoSubfieldLinearCell
from transferfunction import MultiTransferFunction, ScalarTransferFunction
from lnunit import LNUnit, MultiLNUnit    
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y
from kernel import Kernel3D
from movie import Movie, FullFieldFlashMovie
import itertools
import scipy.stats as sps
from make_cell_list import multi_cell_random_generator, make_single_unit_cell_list, make_on_off_cell_list

#Initialize config dicts
single_unit_cell_config = {'number_of_cells':10,
                'lattice_unit_center':(40,30),
                'weights':(.4,-.2),
                'kpeaks': (40,80),
                'amplitude': 10.,
                'sigma':(4,4),
                'width':5}

two_unit_cell_config={'number_of_cells':8,
                      'lattice_unit_center':(40,30),
                      'weights_on':(.4,-.2),
                      'weights_off':(.4,-.1),
                      'kpeaks_on': (40,80),
                      'kpeaks_off': (100,160),
                      'amplitude_on':20.,
                      'amplitude_off':-10.,
                      'sigma_on':(4,4),
                      'sigma_off':(4,4),
                      'subfield_separation':2.,
                      'ang': 45.,
                      'dominant_subunit':'on',
                      'width':5}


all_cell_list = []

#Make Single Unit cells
#======================
sz = [3,6,9]
transient_kpeak_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
sust_kpeak_dist = sps.multivariate_normal(mean=[100., 160.], cov=[[5.0, 0], [0, 5]])

#OFF units
amp_off_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
ncells = [10,5,5]
single_unit_cell_config['amplitude'] = amp_off_dist

#Transient
single_unit_cell_config['kpeaks'] = transient_kpeak_dist
for num_cells, sig in zip(ncells,sz):
    single_unit_cell_config['number_of_cells'] = num_cells
    single_unit_cell_config['sigma'] = (sig,sig)
    all_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config) 
    
#Sustained
single_unit_cell_config['kpeaks'] = sust_kpeak_dist
for num_cells, sig in zip(ncells,sz):
    single_unit_cell_config['number_of_cells'] = num_cells
    single_unit_cell_config['sigma'] = (sig,sig)
    single_unit_cell_config['weights'] = (.4,-.1)
    all_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config)
    
#ON units
amp_on_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
ncells = [5,3,2]
single_unit_cell_config['amplitude'] = amp_on_dist

#Transient
single_unit_cell_config['kpeaks'] = transient_kpeak_dist
for num_cells, sig in zip(ncells,sz):
    single_unit_cell_config['number_of_cells'] = num_cells
    single_unit_cell_config['sigma'] = (sig,sig)
    all_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config) 
    
#Sustained
single_unit_cell_config['kpeaks'] = sust_kpeak_dist
single_unit_cell_config['weights'] = (.4,-.1)
for num_cells, sig in zip(ncells,sz):
    single_unit_cell_config['number_of_cells'] = num_cells
    single_unit_cell_config['sigma'] = (sig,sig)
    all_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config)
    
#Make Two subunit cells
#======================

#Overlapping ON/OFF
ncells = 4
sz = 9
ang_dist = sps.rv_discrete(values=(np.arange(0,180,45), 1./ncells*np.ones((1,ncells))))
amp_on_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
amp_off_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
kpeak_on_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
kpeak_off_dist =  sps.multivariate_normal(mean=[50., 90.], cov=[[5.0, 0], [0, 5]])
wts_on = wts_off = (.4,-.2)
subfield_sep = 2.

two_unit_cell_config['number_of_cells'] = ncells
two_unit_cell_config['ang'] = ang_dist
two_unit_cell_config['amplitude_on'] = amp_on_dist
two_unit_cell_config['amplitude_off'] = amp_off_dist
two_unit_cell_config['kpeaks_on'] = kpeak_on_dist
two_unit_cell_config['kpeaks_off'] = kpeak_off_dist
two_unit_cell_config['weights_on'] = wts_on
two_unit_cell_config['weights_off'] = wts_off
two_unit_cell_config['sigma_on'] = (sz,sz)
two_unit_cell_config['sigma_off'] = (sz,sz)
two_unit_cell_config['subfield_separation'] = subfield_sep

all_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)

#Separate On/Off
ncells = 8
sz = 6
ang_dist = np.arange(0,360,45)
subfield_sep = 4.

two_unit_cell_config['number_of_cells'] = ncells
two_unit_cell_config['ang'] = ang_dist
two_unit_cell_config['sigma_on'] = (sz,sz)
two_unit_cell_config['sigma_off'] = (sz,sz)
two_unit_cell_config['subfield_separation'] = subfield_sep

kpeak_dom_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
kpeak_nondom_dist =  sps.multivariate_normal(mean=[100., 160.], cov=[[5.0, 0], [0, 5]])
wts_dom = (.4,-.2)
wts_nondom = (.4,-.1)

#On-dominant
dom_subunit = 'on'
if dom_subunit=='on':
    two_unit_cell_config['dominant_subunit'] = dom_subunit
    amp_dom_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
    amp_nondom_dist = sps.rv_discrete(values=([-10,-15], [.5,.5]))
    two_unit_cell_config['amplitude_on'] = amp_dom_dist
    two_unit_cell_config['amplitude_off'] = amp_nondom_dist
    two_unit_cell_config['kpeaks_on'] = kpeak_dom_dist
    two_unit_cell_config['kpeaks_off'] = kpeak_nondom_dist
    two_unit_cell_config['weights_on'] = wts_dom
    two_unit_cell_config['weights_off'] = wts_nondom
    all_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)
    
#Off-dominant
dom_subunit = 'off'
if dom_subunit=='off':
    two_unit_cell_config['dominant_subunit'] = dom_subunit
    amp_dom_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
    amp_nondom_dist = sps.rv_discrete(values=([10,15], [.5,.5]))
    two_unit_cell_config['amplitude_off'] = amp_dom_dist
    two_unit_cell_config['amplitude_on'] = amp_nondom_dist
    two_unit_cell_config['kpeaks_off'] = kpeak_dom_dist
    two_unit_cell_config['kpeaks_on'] = kpeak_nondom_dist
    two_unit_cell_config['weights_off'] = wts_dom
    two_unit_cell_config['weights_on'] = wts_nondom
    all_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)
    
    
#print len(all_cell_list)
print all_cell_list[-1].to_dict()

    
sys.exit()    











# movie_file = '/data/mat/iSee_temp_shared/TouchOfEvil_norm.npy'
# m_data = np.load(movie_file, 'r')
# m = Movie(m_data[1000:], frame_rate=30.)
# 
# # Define movie:
# row_values = np.arange(60) # height/vertical extent of movie (degrees (should come out of retinal model))
# col_values = np.arange(80) # width/horizontal extent of movie (degrees) (''')
# transfer_function = ScalarTransferFunction('Heaviside(s)*s')#('s')
# 
# temporal_filter_on = TemporalFilterCosineBump((.4,-.3), (40,80)) # weights and temporal peaks for basis functions
# temporal_filter_off = TemporalFilterCosineBump((.4,-.3), (80,120)) # weights and temporal peaks for basis functions
# 
# cell_list = []
# ctr_start_on = (40,30)
# ctr_start_off = (40,30)
# 
# oncells=make_cell_list(numcells=10,ctr_start=ctr_start_on,sigma_vals=(2.,2.),amplitude=20., kpeaks_wts=(.4,-.3), kpeaks_vals=(40,80),transfer_function=transfer_function)
# offcells=make_cell_list(numcells=10,ctr_start=ctr_start_off,sigma_vals=(2.,2.),amplitude=-20., kpeaks_wts=(.4,-.3), kpeaks_vals=(80,120),transfer_function=transfer_function)
# 
# cell_list = oncells + offcells    
#     
# #Create two-subfield cells/units
# for ii in range(10):
#     xi = ctr_start_off[0] + np.random.uniform(-2.5,2.5)
#     yi = ctr_start_off[1] + np.random.uniform(-2.5,2.5)
#     spatial_filter_on = GaussianSpatialFilter(translate=(xi,yi),sigma=(4.,4.), origin=(0,0)) # all distances measured from BOTTOM LEFT
#     on_spatiotemporal_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter_on, amplitude=20.)
#     #on_lnunit = LNUnit(on_spatiotemporal_filter, transfer_function)
#     spatial_filter_off = GaussianSpatialFilter(translate=(xi,yi),sigma=(2.,2.), origin=(0,0)) # all distances measured from BOTTOM LEFT
#     off_spatiotemporal_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter_off, amplitude=-20.)
#     ang = np.random.random_integers(0,360)
#     cell = TwoSubfieldLinearCell(on_spatiotemporal_filter, off_spatiotemporal_filter, subfield_separation=1, onoff_axis_angle=ang, dominant_subfield_location=(xi,yi))
#     #k = cell.dominant_filter.t_slice(.05,row_values,col_values)
#     #k.imshow()
#     #k = cell.dominant_filter.show_spatial_filter(.05,row_values,col_values)
#     #k.imshow()
#     cell_list.append(cell)
# 
# lgn = LGNModel(cell_list) #Here include a list of all cells
# y = lgn.evaluate(m, downsample=50) #Does the filtering + non-linearity on movie object m
# heat_plot(y, colorbar=False)
# 
# plt.figure()
# plt.plot(y[0][1],'r')
# plt.plot(y[15][1],'b')
# plt.plot(y[29][1],'m')
# plt.show()
# 
# # Create tON cells/units:
# # for ii in range(10):
# #     xi = ctr_start_on[0] + np.random.uniform(-2.5,2.5)
# #     yi = ctr_start_off[1] + np.random.uniform(-2.5,2.5)
# #     spatial_filter_on = GaussianSpatialFilter(translate=(xi,yi),sigma=(2.,2.),  origin=(0,0)) # all distances measured from BOTTOM LEFT
# #     on_spatiotemporal_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter_on,amplitude=20.)
# # #     k = on_spatiotemporal_filter.t_slice(.05,row_values,col_values)
# # #     k.imshow()
# #     cell = LGNOnCell(on_spatiotemporal_filter, transfer_function)
# #     cell_list.append(cell)
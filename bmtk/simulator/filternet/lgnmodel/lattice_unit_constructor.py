import numpy as np
import scipy.stats as sps

from .make_cell_list import multi_cell_random_generator, make_single_unit_cell_list, make_on_off_cell_list


def make_lattice_unit(lattice_unit_center=None):
    cell_list = []
    tON_cell_list = make_tON_cell_list(lattice_unit_center)
    tOFF_cell_list = make_tOFF_cell_list(lattice_unit_center)
    sON_cell_list = make_sON_cell_list(lattice_unit_center)
    sOFF_cell_list = make_sOFF_cell_list(lattice_unit_center)
    overlap_onoff_cell_list = make_overlapping_onoff_cell_list(lattice_unit_center)
    separate_onoff_cell_list = make_separate_onoff_cell_list(lattice_unit_center)
    
    cell_list = tON_cell_list + tOFF_cell_list + sON_cell_list + sOFF_cell_list + overlap_onoff_cell_list + separate_onoff_cell_list 
    
    return cell_list
    

def make_tON_cell_list(lattice_unit_center):
    tON_cell_list = []
    
    single_unit_cell_config = {}
    single_unit_cell_config['lattice_unit_center']=lattice_unit_center
    single_unit_cell_config['width'] = 5.
    sz = [3,6,9]
    ncells = [5,3,2]
    amp_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
    kpeaks_dist =  sps.multivariate_normal(mean=[15., 35.], cov=[[5.0, 0], [0, 5]])
    wts = (4.,-2.5)
    delays = (0.,0.)
    single_unit_cell_config['amplitude'] = amp_dist
    single_unit_cell_config['kpeaks'] = kpeaks_dist
    single_unit_cell_config['weights'] = wts
    single_unit_cell_config['delays'] = delays
    for num_cells, sig in zip(ncells,sz):
        single_unit_cell_config['number_of_cells'] = num_cells
        single_unit_cell_config['sigma'] = (sig,sig)
        tON_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config)
    
    return tON_cell_list

def make_tOFF_cell_list(lattice_unit_center):
    tOFF_cell_list = []
    
    single_unit_cell_config = {}
    single_unit_cell_config['lattice_unit_center']=lattice_unit_center
    single_unit_cell_config['width'] = 5.
    sz = [3,6,9]
    ncells = [10,5,5]
    amp_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
#     kpeaks_dist =  sps.multivariate_normal(mean=[40., 80.], cov=[[5.0, 0], [0, 5]])
#     wts = (.4,-.2)
    kpeaks_dist =  sps.multivariate_normal(mean=[15., 35.], cov=[[5.0, 0], [0, 5]])
    wts = (4.,-2.5)
    delays = (0.,0.)
    single_unit_cell_config['amplitude'] = amp_dist
    single_unit_cell_config['kpeaks'] = kpeaks_dist
    single_unit_cell_config['weights'] = wts
    single_unit_cell_config['delays'] = delays
    for num_cells, sig in zip(ncells,sz):
        single_unit_cell_config['number_of_cells'] = num_cells
        single_unit_cell_config['sigma'] = (sig,sig)
        tOFF_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config) 
    
    #print len(tOFF_cell_list)        
    return tOFF_cell_list

def make_sON_cell_list(lattice_unit_center):
    sON_cell_list = []
    
    single_unit_cell_config = {}
    single_unit_cell_config['lattice_unit_center']=lattice_unit_center
    single_unit_cell_config['width'] = 5.
    sz = [3,6,9]
    ncells = [5,3,2]
    amp_dist = sps.rv_discrete(values=([20,25], [.5, .5]))
    kpeaks_dist =  sps.multivariate_normal(mean=[80., 120.], cov=[[5.0, 0], [0, 5]])
    wts = (4.,-.85)
    delays = (0.,0.)
    single_unit_cell_config['amplitude'] = amp_dist
    single_unit_cell_config['kpeaks'] = kpeaks_dist
    single_unit_cell_config['weights'] = wts
    single_unit_cell_config['delays'] = delays
    for num_cells, sig in zip(ncells,sz):
        single_unit_cell_config['number_of_cells'] = num_cells
        single_unit_cell_config['sigma'] = (sig,sig)
        sON_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config) 
    
    return sON_cell_list


def make_sOFF_cell_list(lattice_unit_center):
    sOFF_cell_list = []
    
    single_unit_cell_config = {}
    single_unit_cell_config['lattice_unit_center']=lattice_unit_center
    single_unit_cell_config['width'] = 5.
    sz = [3,6,9]
    ncells = [10,5,5]
    amp_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
    kpeaks_dist =  sps.multivariate_normal(mean=[80., 120.], cov=[[5.0, 0], [0, 5]])
    wts = (4.,-.85)
    delays = (0.,0.)
    single_unit_cell_config['amplitude'] = amp_dist
    single_unit_cell_config['kpeaks'] = kpeaks_dist
    single_unit_cell_config['weights'] = wts
    single_unit_cell_config['delays'] = delays
    for num_cells, sig in zip(ncells,sz):
        single_unit_cell_config['number_of_cells'] = num_cells
        single_unit_cell_config['sigma'] = (sig,sig)
        sOFF_cell_list += multi_cell_random_generator(make_single_unit_cell_list, **single_unit_cell_config) 
    
    return sOFF_cell_list


def make_overlapping_onoff_cell_list(lattice_unit_center):
    overlap_onoff_cell_list = []
    
    two_unit_cell_config = {}
    two_unit_cell_config['lattice_unit_center']=lattice_unit_center
    two_unit_cell_config['width']=5.
    
    ncells = 4
    sz = 9
    ang_dist = sps.rv_discrete(values=(np.arange(0,180,45), 1./ncells*np.ones(ncells)))
    amp_on_dist = sps.rv_discrete(values=([20,25], [.5,.5]))
    amp_off_dist = sps.rv_discrete(values=([-20,-25], [.5,.5]))
    kpeak_on_dist =  sps.multivariate_normal(mean=[15., 35.], cov=[[5.0, 0], [0, 5]])
    kpeak_off_dist =  sps.multivariate_normal(mean=[20., 40.], cov=[[5.0, 0], [0, 5]])
    wts_on = wts_off = (4.,-2.5)
    delays_on = delays_off = (0.,0.)
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
    two_unit_cell_config['dominant_subunit']='on'
    two_unit_cell_config['delays_on']=delays_on
    two_unit_cell_config['delays_off']=delays_off
    
    overlap_onoff_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)
    
    return overlap_onoff_cell_list


def make_separate_onoff_cell_list(lattice_unit_center):
    separate_onoff_cell_list = []
    
    two_unit_cell_config = {}
    two_unit_cell_config['lattice_unit_center']=lattice_unit_center
    two_unit_cell_config['width']=5.
    
    ncells = 8
    sz = 6
    ang_dist = np.arange(0,360,45)
    subfield_sep = 4.

    kpeak_dom_dist =  sps.multivariate_normal(mean=[15., 35.], cov=[[5.0, 0], [0, 5]])
    kpeak_nondom_dist =  sps.multivariate_normal(mean=[80., 120.], cov=[[5.0, 0], [0, 5]])
    wts_dom = (4.,-2.5)
    wts_nondom = (4,-.85)
    delays_dom = delays_nondom = (0.,0.)    
    
    two_unit_cell_config['number_of_cells'] = ncells
    two_unit_cell_config['ang'] = ang_dist
    two_unit_cell_config['sigma_on'] = (sz,sz)
    two_unit_cell_config['sigma_off'] = (sz,sz)
    two_unit_cell_config['subfield_separation'] = subfield_sep
    
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
        two_unit_cell_config['delays_on'] = delays_dom
        two_unit_cell_config['delays_off'] = delays_nondom
        separate_onoff_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)
        
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
        two_unit_cell_config['delays_off'] = delays_dom
        two_unit_cell_config['delays_on'] = delays_nondom
        separate_onoff_cell_list += multi_cell_random_generator(make_on_off_cell_list, **two_unit_cell_config)
    
    #print len(separate_onoff_cell_list)    
    return separate_onoff_cell_list

if __name__ == "__main__":
    lattice_unit_center = (40,30)
    lattice_cell_list = make_lattice_unit(lattice_unit_center)
    print(len(lattice_cell_list))
        
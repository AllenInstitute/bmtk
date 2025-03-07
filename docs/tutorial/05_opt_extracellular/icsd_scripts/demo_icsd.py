# -*- coding: utf-8 -*-
'''iCSD toolbox demonstration script'''
import matplotlib.pyplot as plt
import numpy as np
import icsd
from scipy import io
import neo
import quantities as pq

#patch quantities with the SI unit Siemens if it does not exist
for symbol, prefix, definition, u_symbol in zip(
    ['siemens', 'S', 'mS', 'uS', 'nS', 'pS'],
    ['', '', 'milli', 'micro', 'nano', 'pico'],
    [pq.A/pq.V, pq.A/pq.V, 'S', 'mS', 'uS', 'nS'],
    [None, None, None, None, u'µS', None]):
    if type(definition) is str:
        definition = lastdefinition / 1000
    if not hasattr(pq, symbol):
        setattr(pq, symbol, pq.UnitQuantity(
            prefix + 'siemens',
            definition,
            symbol=symbol,
            u_symbol=u_symbol))
    lastdefinition = definition

#loading test data
test_data = io.loadmat('test_data.mat')

#prepare lfp data for use, by changing the units to SI and append quantities,
#along with electrode geometry, conductivities and assumed source geometry
lfp_data = test_data['pot1'] * 1E-6 * pq.V        # [uV] -> [V]
z_data = np.linspace(100E-6, 2300E-6, 23) * pq.m  # [m]
diam = 500E-6 * pq.m                              # [m]
h = 100E-6 * pq.m                                 # [m]
sigma = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)]
sigma_top = 0.3 * pq.S / pq.m                     # [S/m] or [1/(ohm*m)]

# Input dictionaries for each method
delta_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,          # source diameter
    'sigma' : sigma,        # extracellular conductivity
    'sigma_top' : sigma,    # conductivity on top of cortex
    'f_type' : 'gaussian',  # gaussian filter
    'f_order' : (3, 1),     # 3-point filter, sigma = 1.
}
step_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,
    'h' : h,                # source thickness
    'sigma' : sigma,
    'sigma_top' : sigma,
    'tol' : 1E-12,          # Tolerance in numerical integration
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
}
spline_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'diam' : diam,
    'sigma' : sigma,
    'sigma_top' : sigma,
    'num_steps' : 201,      # Spatial CSD upsampling to N steps
    'tol' : 1E-12,
    'f_type' : 'gaussian',
    'f_order' : (20, 5),
}
std_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'sigma' : sigma,
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
}


#Create the different CSD-method class instances. We use the class methods
#get_csd() and filter_csd() below to get the raw and spatially filtered
#versions of the current-source density estimates.
csd_dict = dict(
    delta_icsd = icsd.DeltaiCSD(**delta_input),
    step_icsd = icsd.StepiCSD(**step_input),
    spline_icsd = icsd.SplineiCSD(**spline_input),
    std_csd = icsd.StandardCSD(**std_input), 
)

#plot
for method, csd_obj in list(csd_dict.items()):
    fig, axes = plt.subplots(3,1, figsize=(8,8))
    
    #plot LFP signal
    ax = axes[0]
    im = ax.imshow(np.array(lfp_data), origin='upper', vmin=-abs(lfp_data).max(), \
              vmax=abs(lfp_data).max(), cmap='jet_r', interpolation='nearest')
    ax.axis(ax.axis('tight'))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('LFP (%s)' % lfp_data.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_title('LFP')
    ax.set_ylabel('ch #')

    #plot raw csd estimate
    csd = csd_obj.get_csd()
    ax = axes[1]
    im = ax.imshow(np.array(csd), origin='upper', vmin=-abs(csd).max(), \
          vmax=abs(csd).max(), cmap='jet_r', interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.set_title(csd_obj.name)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('CSD (%s)' % csd.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_ylabel('ch #')
    
    #plot spatially filtered csd estimate
    ax = axes[2]
    csd = csd_obj.filter_csd(csd)
    im = ax.imshow(np.array(csd), origin='upper', vmin=-abs(csd).max(), \
          vmax=abs(csd).max(), cmap='jet_r', interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.set_title(csd_obj.name + ', filtered')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('CSD (%s)' % csd.dimensionality.string)
    ax.set_ylabel('ch #')
    ax.set_xlabel('timestep')


# ############################################################################ #
# Demonstrate icsd.estimate_csd() function on neo.AnalogSignalArray object     #
# ############################################################################ #
lfp_data = neo.AnalogSignalArray(lfp_data.T, sampling_rate=2.*pq.kHz)

# Input dictionaries for each method
delta_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'delta',
    'diam' : diam,          # source diameter
    'sigma' : sigma,        # extracellular conductivity
    'sigma_top' : sigma,    # conductivity on top of cortex
    'f_type' : 'gaussian',  # gaussian filter
    'f_order' : (3, 1),     # 3-point filter, sigma = 1.
}
step_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'step',
    'diam' : diam,
    'h' : h,                # source thickness
    'sigma' : sigma,
    'sigma_top' : sigma,
    'tol' : 1E-12,          # Tolerance in numerical integration
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
}
spline_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'spline',
    'diam' : diam,
    'sigma' : sigma,
    'sigma_top' : sigma,
    'num_steps' : 201,      # Spatial CSD upsampling to N steps
    'tol' : 1E-12,
    'f_type' : 'gaussian',
    'f_order' : (20, 5),
}
std_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'standard',
    'sigma' : sigma,
    'f_type' : 'gaussian',
    'f_order' : (3, 1),
}

#compute CSD and filtered CSD estimates. Note that the returned argument of the
#function is a tuple of neo.AnalogSignalArray objects (csd, csd_filtered)
csd_dict = dict(
    delta_icsd = icsd.estimate_csd(**delta_input),
    step_icsd = icsd.estimate_csd(**step_input),
    spline_icsd = icsd.estimate_csd(**spline_input),
    std_csd = icsd.estimate_csd(**std_input), 
)

#plot
for method, csd_obj in list(csd_dict.items()):
    fig, axes = plt.subplots(3,1, figsize=(8,8))
    
    #plot LFP signal
    ax = axes[0]
    im = ax.imshow(lfp_data.magnitude.T, origin='upper',
                   vmin=-abs(lfp_data.magnitude).max(), 
                   vmax=abs(lfp_data.magnitude).max(), cmap='jet_r',
                   interpolation='nearest')
    ax.axis(ax.axis('tight'))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('LFP (%s)' % lfp_data.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_title('LFP')
    ax.set_ylabel('ch #')

    #plot raw csd estimate
    csd = csd_obj[0]
    ax = axes[1]
    im = ax.imshow(csd.magnitude.T, origin='upper',
                   vmin=-abs(csd.magnitude).max(), 
                   vmax=abs(csd.magnitude).max(), cmap='jet_r',
                   interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.set_title(method)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('CSD (%s)' % csd.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_ylabel('ch #')
    
    #plot spatially filtered csd estimate
    ax = axes[2]
    csd = csd_obj[1]
    im = ax.imshow(csd.magnitude.T, origin='upper',
                   vmin=-abs(csd.magnitude).max(), 
                   vmax=abs(csd.magnitude).max(), cmap='jet_r',
                   interpolation='nearest')
    ax.axis(ax.axis('tight'))
    ax.set_title(method + ', filtered')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('CSD (%s)' % csd.dimensionality.string)
    ax.set_ylabel('ch #')
    ax.set_xlabel('timestep')



plt.show()

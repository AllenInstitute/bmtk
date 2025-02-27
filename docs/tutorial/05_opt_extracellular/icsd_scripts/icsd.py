# -*- coding: utf-8 -*-
"""
py-iCSD toolbox!
Translation of the core functionality of the CSDplotter MATLAB package
to python.

The methods were originally developed by Klas H. Pettersen, as described in:
Klas H. Pettersen, Anna Devor, Istvan Ulbert, Anders M. Dale, Gaute T. Einevoll,
Current-source density estimation based on inversion of electrostatic forward
solution: Effects of finite extent of neuronal activity and conductivity
discontinuities, Journal of Neuroscience Methods, Volume 154, Issues 1-2,
30 June 2006, Pages 116-133, ISSN 0165-0270,
http://dx.doi.org/10.1016/j.jneumeth.2005.12.005.
(http://www.sciencedirect.com/science/article/pii/S0165027005004541)

The method themselves are implemented as callable subclasses of the base
CSD class object, which sets some common attributes,
and a basic function for calculating the iCSD, and a generic spatial filter
implementation.

The raw- and filtered CSD estimates are returned as Quantity arrays.

Requires pylab environment to work, i.e numpy+scipy+matplotlib, with the
addition of quantities (http://pythonhosted.org/quantities) and
neo (https://pythonhosted.org/neo)-

Original implementation from CSDplotter-0.1.1
(http://software.incf.org/software/csdplotter) by Klas. H. Pettersen 2005.

Written by:
- Espen.Hagen@umb.no, 2010,
- e.hagen@fz-juelich.de, 2015-2016


Example
-------
#!/usr/env/python

import matplotlib.pyplot as plt
import numpy as np
import icsd
from scipy import io
import quantities as pq

#loading test data
test_data = io.loadmat('test_data.mat')

#prepare lfp data for use, by changing the units to SI and append quantities,
#along with electrode geometry and conductivities
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
    'h' : h,
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
for method, csd_obj in csd_dict.items():
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

plt.show()


Example
-------
#!/usr/env/python

import matplotlib.pyplot as plt
import numpy as np
import icsd
from scipy import io
import neo
import quantities as pq

#loading test data
test_data = io.loadmat('test_data.mat')

#prepare lfp data for use, by changing the units to SI and append quantities,
#along with electrode geometry and conductivities
lfp_data = neo.AnalogSignalArray(test_data['pot1'].T * 1E-6 * pq.V,
                                 sampling_rate=2.*pq.kHz)
z_data = np.linspace(100E-6, 2300E-6, 23) * pq.m  # [m]
diam = 500E-6 * pq.m                              # [m]
h = 100E-6 * pq.m                                 # [m]
sigma = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)]
sigma_top = 0.3 * pq.S / pq.m                     # [S/m] or [1/(ohm*m)]

# Input dictionaries for each method
delta_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'delta',
    'diam' : diam,        # source diameter
    'sigma' : sigma,           # extracellular conductivity
    'sigma_top' : sigma,       # conductivity on top of cortex
    'f_type' : 'gaussian',  # gaussian filter
    'f_order' : (3, 1),     # 3-point filter, sigma = 1.
}
step_input = {
    'lfp' : lfp_data,
    'coord_electrode' : z_data,
    'method' : 'step',
    'diam' : diam,
    'h' : h,
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
for method, csd_obj in csd_dict.items():
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



"""

import numpy as np
import scipy.integrate as si
import scipy.signal as ss
import quantities as pq
import neo


# patch quantities with the SI unit Siemens if it does not exist
for symbol, prefix, definition, u_symbol in zip(
    ["siemens", "S", "mS", "uS", "nS", "pS"],
    ["", "", "milli", "micro", "nano", "pico"],
    [pq.A / pq.V, pq.A / pq.V, "S", "mS", "uS", "nS"],
    [None, None, None, None, "µS", None],
):
    if type(definition) is str:
        definition = lastdefinition / 1000
    if not hasattr(pq, symbol):
        setattr(
            pq,
            symbol,
            pq.UnitQuantity(
                prefix + "siemens", definition, symbol=symbol, u_symbol=u_symbol
            ),
        )
    lastdefinition = definition


class CSD(object):
    """Base iCSD class"""

    def __init__(self, lfp, f_type="gaussian", f_order=(3, 1)):
        """Initialize parent class iCSD

        Parameters
        ----------
        lfp : np.ndarray * quantity.Quantity
            LFP signal of shape (# channels, # time steps)
        f_type : str
            type of spatial filter, must be a scipy.signal filter design method
        f_order : list
            settings for spatial filter, arg passed to  filter design function
        """
        self.name = "CSD estimate parent class"
        self.lfp = lfp
        self.f_matrix = np.eye(lfp.shape[0]) * pq.m**3 / pq.S
        self.f_type = f_type
        self.f_order = f_order

    def get_csd(
        self,
    ):
        """
        Perform the CSD estimate from the LFP and forward matrix F, i.e as
        CSD=F**-1*LFP

        Arguments
        ---------

        Returns
        -------
        csd : np.ndarray * quantity.Quantity
            Array with the csd estimate
        """
        csd = np.linalg.solve(self.f_matrix, self.lfp)

        return csd * (self.f_matrix.units**-1 * self.lfp.units).simplified

    def filter_csd(self, csd, filterfunction="convolve"):
        """
        Spatial filtering of the CSD estimate, using an N-point filter

        Arguments
        ---------
        csd : np.ndarrray * quantity.Quantity
            Array with the csd estimate
        filterfunction : str
            'filtfilt' or 'convolve'. Apply spatial filter using
            scipy.signal.filtfilt or scipy.signal.convolve.
        """
        if self.f_type == "gaussian":
            try:
                assert len(self.f_order) == 2
            except AssertionError as ae:
                raise ae("filter order f_order must be a tuple of length 2")
        else:
            try:
                assert self.f_order > 0 and isinstance(self.f_order, int)
            except AssertionError as ae:
                raise ae("Filter order must be int > 0!")
        try:
            assert filterfunction in ["filtfilt", "convolve"]
        except AssertionError as ae:
            raise ae("{} not equal to 'filtfilt' or 'convolve'".format(filterfunction))

        if self.f_type == "boxcar":
            num = ss.windows.boxcar(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == "hamming":
            num = ss.windows.hamming(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == "triangular":
            num = ss.windows.triang(self.f_order)
            denom = np.array([num.sum()])
        elif self.f_type == "gaussian":
            num = ss.windows.gaussian(self.f_order[0], self.f_order[1])
            denom = np.array([num.sum()])
        elif self.f_type == "identity":
            num = np.array([1.0])
            denom = np.array([1.0])
        else:
            print("%s Wrong filter type!" % self.f_type)
            raise

        num_string = "[ "
        for i in num:
            num_string = num_string + "%.3f " % i
        num_string = num_string + "]"
        denom_string = "[ "
        for i in denom:
            denom_string = denom_string + "%.3f " % i
        denom_string = denom_string + "]"

        print(
            (
                "discrete filter coefficients: \nb = {}, \na = {}".format(
                    num_string, denom_string
                )
            )
        )

        if filterfunction == "filtfilt":
            return ss.filtfilt(num, denom, csd, axis=0) * csd.units
        elif filterfunction == "convolve":
            csdf = csd / csd.units
            for i in range(csdf.shape[1]):
                csdf[:, i] = ss.convolve(csdf[:, i], num / denom.sum(), "same")
            return csdf * csd.units


class StandardCSD(CSD):
    """
    Standard CSD method with and without Vaknin electrodes
    """

    def __init__(
        self,
        lfp,
        coord_electrode=np.linspace(0, 1400e-6, 15) * pq.m,
        sigma=0.3 * pq.S / pq.m,
        vaknin_el=True,
        f_type="gaussian",
        f_order=(3, 1),
    ):
        """
        Initialize standard CSD method class with and without Vaknin electrodes.

        Parameters
        ----------
        lfp : np.ndarray * quantity.Quantity
            LFP signal of shape (# channels, # time steps) in units of V
        coord_electrode : np.ndarray * quantity.Quantity
            depth of evenly spaced electrode contact points of shape
            (# contacts, ) in units of m, must be monotonously increasing
        sigma : float * quantity.Quantity
            conductivity of tissue in units of S/m or 1/(ohm*m)
        vaknin_el : bool
            flag for using method of Vaknin to endpoint electrodes
        f_type : str
            type of spatial filter, must be a scipy.signal filter design method
        f_order : list
            settings for spatial filter, arg passed to  filter design function
        """
        CSD.__init__(self, lfp, f_type, f_order)

        self.name = "Standard CSD method"
        self.coord_electrode = coord_electrode
        self.sigma = sigma
        self.vaknin_el = vaknin_el

        try:
            assert np.all(np.diff(np.diff(coord_electrode))) == 0
        except AssertionError as ae:
            print("coord_electrode not monotonously varying")
            raise ae

        if vaknin_el:
            # extend array of lfps by duplicating potential at endpoint contacts
            if lfp.ndim == 1:
                self.lfp = np.empty((lfp.shape[0] + 2,)) * lfp.units
            else:
                self.lfp = np.empty((lfp.shape[0] + 2, lfp.shape[1])) * lfp.units
            self.lfp[0,] = lfp[0,]
            self.lfp[1:-1,] = lfp
            self.lfp[-1,] = lfp[-1,]
        else:
            self.lfp = lfp

        self.f_inv_matrix = self.get_f_inv_matrix()

    def get_f_inv_matrix(self):
        """Calculate the inverse F-matrix for the standard CSD method"""
        h_val = abs(np.diff(self.coord_electrode)[0])

        f_inv = -np.eye(self.lfp.shape[0])

        # Inner matrix elements  is just the discrete laplacian coefficients
        for j in range(1, f_inv.shape[0] - 1):
            f_inv[j, j - 1 : j + 2] = np.array([1.0, -2.0, 1.0])

        return f_inv * -self.sigma / h_val

    def get_csd(self):
        """
        Perform the iCSD calculation, i.e: iCSD=F_inv*LFP

        Returns
        -------
        csd : np.ndarray * quantity.Quantity
            Array with the csd estimate
        """
        csd = np.dot(self.f_inv_matrix, self.lfp)[1:-1,]
        # `np.dot()` does not return correct units, so the units of `csd` must
        # be assigned manually
        csd_units = (self.f_inv_matrix.units * self.lfp.units).simplified
        csd = csd.magnitude * csd_units

        return csd


class DeltaiCSD(CSD):
    """
    delta-iCSD method
    """

    def __init__(
        self,
        lfp,
        coord_electrode=np.linspace(0, 1400e-6, 15) * pq.m,
        diam=500e-6 * pq.m,
        sigma=0.3 * pq.S / pq.m,
        sigma_top=0.3 * pq.S / pq.m,
        f_type="gaussian",
        f_order=(3, 1),
    ):
        """
        Initialize the delta-iCSD method class object

        Parameters
        ----------
        lfp : np.ndarray * quantity.Quantity
            LFP signal of shape (# channels, # time steps) in units of V
        coord_electrode : np.ndarray * quantity.Quantity
            depth of evenly spaced electrode contact points of shape
            (# contacts, ) in units of m
        diam : float * quantity.Quantity
            diamater of the assumed circular planar current sources centered
            at each contact
        sigma : float * quantity.Quantity
            conductivity of tissue in units of S/m or 1/(ohm*m)
        sigma_top : float * quantity.Quantity
            conductivity on top of tissue in units of S/m or 1/(ohm*m)
        f_type : str
            type of spatial filter, must be a scipy.signal filter design method
        f_order : list
            settings for spatial filter, arg passed to  filter design function

        """
        CSD.__init__(self, lfp)

        try:
            assert diam.units == coord_electrode.units
        except AssertionError as ae:
            print(
                "units of coord_electrode ({}) and diam ({}) differ".format(
                    coord_electrode.units, diam.units
                )
            )
            raise ae

        try:
            assert np.all(np.diff(coord_electrode) > 0)
        except AssertionError as ae:
            print("values of coord_electrode not continously increasing")
            raise ae

        try:
            assert diam.size == 1 or diam.size == coord_electrode.size
            if diam.size == coord_electrode.size:
                assert np.all(diam > 0 * diam.units)
            else:
                assert diam > 0 * diam.units
        except AssertionError as ae:
            print("diam must be positive scalar or of same shape as coord_electrode")
            raise ae
        if diam.size == 1:
            diam = np.ones(coord_electrode.size) * diam

        self.name = "delta-iCSD method"
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.sigma = sigma
        self.sigma_top = sigma_top
        self.f_type = f_type
        self.f_order = f_order

        # initialize F- and iCSD-matrices
        self.f_matrix = np.empty((self.coord_electrode.size, self.coord_electrode.size))

        self.f_matrix = self.get_f_matrix()

    def get_f_matrix(self):
        """Calculate the F-matrix"""
        f_matrix = (
            np.empty((self.coord_electrode.size, self.coord_electrode.size))
            * self.coord_electrode.units
        )
        for j in range(self.coord_electrode.size):
            for i in range(self.coord_electrode.size):
                f_matrix[j, i] = (
                    np.sqrt(
                        (self.coord_electrode[j] - self.coord_electrode[i]) ** 2
                        + (self.diam[j] / 2) ** 2
                    )
                    - abs(self.coord_electrode[j] - self.coord_electrode[i])
                ) + (self.sigma - self.sigma_top) / (self.sigma + self.sigma_top) * (
                    np.sqrt(
                        (self.coord_electrode[j] + self.coord_electrode[i]) ** 2
                        + (self.diam[j] / 2) ** 2
                    )
                    - abs(self.coord_electrode[j] + self.coord_electrode[i])
                )

        f_matrix /= 2 * self.sigma
        return f_matrix


class StepiCSD(CSD):
    """step-iCSD method"""

    def __init__(
        self,
        lfp,
        coord_electrode=np.linspace(0, 1400e-6, 15) * pq.m,
        diam=500e-6 * pq.m,
        h=np.ones(15) * 100e-6 * pq.m,
        sigma=0.3 * pq.S / pq.m,
        sigma_top=0.3 * pq.S / pq.m,
        tol=1e-6,
        f_type="gaussian",
        f_order=(3, 1),
    ):
        """
        Initializing step-iCSD method class object

        Parameters
        ----------
        lfp : np.ndarray * quantity.Quantity
            LFP signal of shape (# channels, # time steps) in units of V
        coord_electrode : np.ndarray * quantity.Quantity
            depth of evenly spaced electrode contact points of shape
            (# contacts, ) in units of m
        diam : float or np.ndarray * quantity.Quantity
            diameter(s) of the assumed circular planar current sources centered
            at each contact
        h : float or np.ndarray * quantity.Quantity
            assumed thickness of the source cylinders at all or each contact
        sigma : float * quantity.Quantity
            conductivity of tissue in units of S/m or 1/(ohm*m)
        sigma_top : float * quantity.Quantity
            conductivity on top of tissue in units of S/m or 1/(ohm*m)
        tol : float
            tolerance of numerical integration
        f_type : str
            type of spatial filter, must be a scipy.signal filter design method
        f_order : list
            settings for spatial filter, arg passed to  filter design function

        """
        CSD.__init__(self, lfp, f_type, f_order)

        try:
            assert diam.units == coord_electrode.units
        except AssertionError as ae:
            print(
                "units of coord_electrode ({}) and diam ({}) differ".format(
                    coord_electrode.units, diam.units
                )
            )
            raise ae
        try:
            assert np.all(np.diff(coord_electrode) > 0)
        except AssertionError as ae:
            print("values of coord_electrode not continously increasing")
            raise ae

        try:
            assert diam.size == 1 or diam.size == coord_electrode.size
            if diam.size == coord_electrode.size:
                assert np.all(diam > 0 * diam.units)
            else:
                assert diam > 0 * diam.units
        except AssertionError as ae:
            print("diam must be positive scalar or of same shape as coord_electrode")
            raise ae
        if diam.size == 1:
            diam = np.ones(coord_electrode.size) * diam

        try:
            assert h.size == 1 or h.size == coord_electrode.size
            if h.size == coord_electrode.size:
                assert np.all(h > 0 * h.units)
        except AssertionError as ae:
            print("h must be scalar or of same shape as coord_electrode")
            raise ae
        if h.size == 1:
            h = np.ones(coord_electrode.size) * h

        self.name = "step-iCSD method"
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.h = h
        self.sigma = sigma
        self.sigma_top = sigma_top
        self.tol = tol

        # compute forward-solution matrix
        self.f_matrix = self.get_f_matrix()

    def get_f_matrix(self):
        """Calculate F-matrix for step iCSD method"""
        el_len = self.coord_electrode.size
        f_matrix = np.zeros((el_len, el_len))
        for j in range(el_len):
            for i in range(el_len):
                lower_int = self.coord_electrode[i] - self.h[j] / 2
                if lower_int < 0:
                    lower_int = self.h[j].units
                upper_int = self.coord_electrode[i] + self.h[j] / 2

                # components of f_matrix object
                f_cyl0 = si.quad(
                    self._f_cylinder,
                    a=lower_int,
                    b=upper_int,
                    args=(
                        float(self.coord_electrode[j]),
                        float(self.diam[j]),
                        float(self.sigma),
                    ),
                    epsabs=self.tol,
                )[0]
                f_cyl1 = si.quad(
                    self._f_cylinder,
                    a=lower_int,
                    b=upper_int,
                    args=(
                        -float(self.coord_electrode[j]),
                        float(self.diam[j]),
                        float(self.sigma),
                    ),
                    epsabs=self.tol,
                )[0]

                # method of images coefficient
                mom = (self.sigma - self.sigma_top) / (self.sigma + self.sigma_top)

                f_matrix[j, i] = f_cyl0 + mom * f_cyl1

        # assume si.quad trash the units
        return f_matrix * self.h.units**2 / self.sigma.units

    def _f_cylinder(self, zeta, z_val, diam, sigma):
        """function used by class method"""
        f_cyl = (
            1.0
            / (2.0 * sigma)
            * (np.sqrt((diam / 2) ** 2 + ((z_val - zeta)) ** 2) - abs(z_val - zeta))
        )
        return f_cyl


class SplineiCSD(CSD):
    """spline iCSD method"""

    def __init__(
        self,
        lfp,
        coord_electrode=np.linspace(0, 1400e-6, 15) * pq.m,
        diam=500e-6 * pq.m,
        sigma=0.3 * pq.S / pq.m,
        sigma_top=0.3 * pq.S / pq.m,
        tol=1e-6,
        f_type="gaussian",
        f_order=(3, 1),
        num_steps=200,
    ):
        """
        Initializing spline-iCSD method class object

        Parameters
        ----------
        lfp : np.ndarray * quantity.Quantity
            LFP signal of shape (# channels, # time steps) in units of V
        coord_electrode : np.ndarray * quantity.Quantity
            depth of evenly spaced electrode contact points of shape
            (# contacts, ) in units of m
        diam : float * quantity.Quantity
            diamater of the assumed circular planar current sources centered
            at each contact
        sigma : float * quantity.Quantity
            conductivity of tissue in units of S/m or 1/(ohm*m)
        sigma_top : float * quantity.Quantity
            conductivity on top of tissue in units of S/m or 1/(ohm*m)
        tol : float
            tolerance of numerical integration
        f_type : str
            type of spatial filter, must be a scipy.signal filter design method
        f_order : list
            settings for spatial filter, arg passed to  filter design function
        num_steps : int
            number of data points for the spatially upsampled LFP/CSD data

        """
        CSD.__init__(self, lfp, f_type, f_order)

        try:
            assert diam.units == coord_electrode.units
        except AssertionError as ae:
            print(
                "units of coord_electrode ({}) and diam ({}) differ".format(
                    coord_electrode.units, diam.units
                )
            )
            raise
        try:
            assert np.all(np.diff(coord_electrode) > 0)
        except AssertionError as ae:
            print("values of coord_electrode not continously increasing")
            raise ae

        try:
            assert diam.size == 1 or diam.size == coord_electrode.size
            if diam.size == coord_electrode.size:
                assert np.all(diam > 0 * diam.units)
        except AssertionError as ae:
            print("diam must be scalar or of same shape as coord_electrode")
            raise ae
        if diam.size == 1:
            diam = np.ones(coord_electrode.size) * diam

        self.name = "spline-iCSD method"
        self.coord_electrode = coord_electrode
        self.diam = diam
        self.sigma = sigma
        self.sigma_top = sigma_top
        self.tol = tol
        self.num_steps = num_steps

        # compute stuff
        self.f_matrix = self.get_f_matrix()

    def get_f_matrix(self):
        """Calculate the F-matrix for cubic spline iCSD method"""
        el_len = self.coord_electrode.size
        z_js = np.zeros(el_len + 1)
        z_js[:-1] = np.array(self.coord_electrode)
        z_js[-1] = z_js[-2] + float(np.diff(self.coord_electrode).mean())

        # Define integration matrixes
        f_mat0 = np.zeros((el_len, el_len + 1))
        f_mat1 = np.zeros((el_len, el_len + 1))
        f_mat2 = np.zeros((el_len, el_len + 1))
        f_mat3 = np.zeros((el_len, el_len + 1))

        # Calc. elements
        for j in range(el_len):
            for i in range(el_len):
                f_mat0[j, i] = si.quad(
                    self._f_mat0,
                    a=z_js[i],
                    b=z_js[i + 1],
                    args=(z_js[j + 1], float(self.sigma), float(self.diam[j])),
                    epsabs=self.tol,
                )[0]
                f_mat1[j, i] = si.quad(
                    self._f_mat1,
                    a=z_js[i],
                    b=z_js[i + 1],
                    args=(z_js[j + 1], z_js[i], float(self.sigma), float(self.diam[j])),
                    epsabs=self.tol,
                )[0]
                f_mat2[j, i] = si.quad(
                    self._f_mat2,
                    a=z_js[i],
                    b=z_js[i + 1],
                    args=(z_js[j + 1], z_js[i], float(self.sigma), float(self.diam[j])),
                    epsabs=self.tol,
                )[0]
                f_mat3[j, i] = si.quad(
                    self._f_mat3,
                    a=z_js[i],
                    b=z_js[i + 1],
                    args=(z_js[j + 1], z_js[i], float(self.sigma), float(self.diam[j])),
                    epsabs=self.tol,
                )[0]

                # image technique if conductivity not constant:
                if self.sigma != self.sigma_top:
                    f_mat0[j, i] = (
                        f_mat0[j, i]
                        + (self.sigma - self.sigma_top)
                        / (self.sigma + self.sigma_top)
                        * si.quad(
                            self._f_mat0,
                            a=z_js[i],
                            b=z_js[i + 1],
                            args=(-z_js[j + 1], float(self.sigma), float(self.diam[j])),
                            epsabs=self.tol,
                        )[0]
                    )
                    f_mat1[j, i] = (
                        f_mat1[j, i]
                        + (self.sigma - self.sigma_top)
                        / (self.sigma + self.sigma_top)
                        * si.quad(
                            self.f_mat1,
                            a=z_js[i],
                            b=z_js[i + 1],
                            args=(
                                -z_js[j + 1],
                                z_js[i],
                                float(self.sigma),
                                float(self.diam[j]),
                            ),
                            epsabs=self.tol,
                        )[0]
                    )
                    f_mat2[j, i] = (
                        f_mat2[j, i]
                        + (self.sigma - self.sigma_top)
                        / (self.sigma + self.sigma_top)
                        * si.quad(
                            self._f_mat2,
                            a=z_js[i],
                            b=z_js[i + 1],
                            args=(
                                -z_js[j + 1],
                                z_js[i],
                                float(self.sigma),
                                float(self.diam[j]),
                            ),
                            epsabs=self.tol,
                        )[0]
                    )
                    f_mat3[j, i] = (
                        f_mat3[j, i]
                        + (self.sigma - self.sigma_top)
                        / (self.sigma + self.sigma_top)
                        * si.quad(
                            self._f_mat3,
                            a=z_js[i],
                            b=z_js[i + 1],
                            args=(
                                -z_js[j + 1],
                                z_js[i],
                                float(self.sigma),
                                float(self.diam[j]),
                            ),
                            epsabs=self.tol,
                        )[0]
                    )

        e_mat0, e_mat1, e_mat2, e_mat3 = self._calc_e_matrices()

        # Calculate the F-matrix
        f_matrix = np.eye(el_len + 2)
        f_matrix[1:-1, :] = (
            np.dot(f_mat0, e_mat0)
            + np.dot(f_mat1, e_mat1)
            + np.dot(f_mat2, e_mat2)
            + np.dot(f_mat3, e_mat3)
        )

        return f_matrix * self.coord_electrode.units**2 / self.sigma.units

    def get_csd(self):
        """
        Calculate the iCSD using the spline iCSD method

        Returns
        -------
        csd : np.ndarray * quantity.Quantity
            Array with csd estimate


        """
        e_mat = self._calc_e_matrices()

        el_len = self.coord_electrode.size

        # padding the lfp with zeros on top/bottom
        if self.lfp.ndim == 1:
            cs_lfp = np.r_[[0], np.asarray(self.lfp), [0]].reshape(1, -1).T
            csd = np.zeros(self.num_steps)
        else:
            cs_lfp = np.vstack(
                (
                    np.zeros(self.lfp.shape[1]),
                    np.asarray(self.lfp),
                    np.zeros(self.lfp.shape[1]),
                )
            )
            csd = np.zeros((self.num_steps, self.lfp.shape[1]))
        cs_lfp *= self.lfp.units

        # CSD coefficients
        csd_coeff = np.linalg.solve(self.f_matrix, cs_lfp)

        # The cubic spline polynomial coefficients
        a_mat0 = np.dot(e_mat[0], csd_coeff)
        a_mat1 = np.dot(e_mat[1], csd_coeff)
        a_mat2 = np.dot(e_mat[2], csd_coeff)
        a_mat3 = np.dot(e_mat[3], csd_coeff)

        # Extend electrode coordinates in both end by min contact interdistance
        h = np.diff(self.coord_electrode).min()
        z_js = np.zeros(el_len + 2)
        z_js[0] = self.coord_electrode[0] - h
        z_js[1:-1] = self.coord_electrode
        z_js[-1] = self.coord_electrode[-1] + h

        # create high res spatial grid
        out_zs = np.linspace(z_js[1], z_js[-2], self.num_steps)

        # Calculate iCSD estimate on grid from polynomial coefficients.
        i = 0
        for j in range(self.num_steps):
            if out_zs[j] >= z_js[i + 1]:
                i += 1
            csd[j,] = (
                a_mat0[i, :]
                + a_mat1[i, :] * (out_zs[j] - z_js[i])
                + a_mat2[i, :] * (out_zs[j] - z_js[i]) ** 2
                + a_mat3[i, :] * (out_zs[j] - z_js[i]) ** 3
            )

        csd_unit = (self.f_matrix.units**-1 * self.lfp.units).simplified

        return csd * csd_unit

    def _f_mat0(self, zeta, z_val, sigma, diam):
        """0'th order potential function"""
        return (
            1.0
            / (2.0 * sigma)
            * (np.sqrt((diam / 2) ** 2 + ((z_val - zeta)) ** 2) - abs(z_val - zeta))
        )

    def _f_mat1(self, zeta, z_val, zi_val, sigma, diam):
        """1'th order potential function"""
        return (zeta - zi_val) * self._f_mat0(zeta, z_val, sigma, diam)

    def _f_mat2(self, zeta, z_val, zi_val, sigma, diam):
        """2'nd order potential function"""
        return (zeta - zi_val) ** 2 * self._f_mat0(zeta, z_val, sigma, diam)

    def _f_mat3(self, zeta, z_val, zi_val, sigma, diam):
        """3'rd order potential function"""
        return (zeta - zi_val) ** 3 * self._f_mat0(zeta, z_val, sigma, diam)

    def _calc_k_matrix(self):
        """Calculate the K-matrix used by to calculate E-matrices"""
        el_len = self.coord_electrode.size
        h = float(np.diff(self.coord_electrode).min())

        c_jm1 = np.eye(el_len + 2, k=0) / h
        c_jm1[0, 0] = 0

        c_j0 = np.eye(el_len + 2) / h
        c_j0[-1, -1] = 0

        c_jall = c_j0
        c_jall[0, 0] = 1
        c_jall[-1, -1] = 1

        tjp1 = np.eye(el_len + 2, k=1)
        tjm1 = np.eye(el_len + 2, k=-1)

        tj0 = np.eye(el_len + 2)
        tj0[0, 0] = 0
        tj0[-1, -1] = 0

        # Defining K-matrix used to calculate e_mat1-3
        return np.dot(
            np.linalg.inv(
                np.dot(c_jm1, tjm1)
                + 2 * np.dot(c_jm1, tj0)
                + 2 * c_jall
                + np.dot(c_j0, tjp1)
            ),
            3
            * (
                np.dot(np.dot(c_jm1, c_jm1), tj0)
                - np.dot(np.dot(c_jm1, c_jm1), tjm1)
                + np.dot(np.dot(c_j0, c_j0), tjp1)
                - np.dot(np.dot(c_j0, c_j0), tj0)
            ),
        )

    def _calc_e_matrices(self):
        """Calculate the E-matrices used by cubic spline iCSD method"""
        el_len = self.coord_electrode.size
        ## expanding electrode grid
        h = float(np.diff(self.coord_electrode).min())

        ## Define transformation matrices
        c_mat3 = np.eye(el_len + 1) / h

        # Get K-matrix
        k_matrix = self._calc_k_matrix()

        # Define matrixes for C to A transformation:
        tja = np.eye(el_len + 2)[:-1,]
        tjp1a = np.eye(el_len + 2, k=1)[:-1,]

        # Define spline coefficients
        e_mat0 = tja
        e_mat1 = np.dot(tja, k_matrix)
        e_mat2 = 3 * np.dot(c_mat3**2, (tjp1a - tja)) - np.dot(
            np.dot(c_mat3, (tjp1a + 2 * tja)), k_matrix
        )
        e_mat3 = 2 * np.dot(c_mat3**3, (tja - tjp1a)) + np.dot(
            np.dot(c_mat3**2, (tjp1a + tja)), k_matrix
        )

        return e_mat0, e_mat1, e_mat2, e_mat3


def estimate_csd(
    lfp,
    coord_electrode,
    sigma,
    method="standard",
    diam=None,
    h=None,
    sigma_top=None,
    tol=1e-6,
    num_steps=200,
    f_type="identity",
    f_order=None,
):
    """
    Estimates current source density (CSD) from local field potential (LFP)
    recordings from multiple depths of the cortex.

    Parameters
    ----------
    lfp : neo.AnalogSignalArray
        LFP signals from which CSD is estimated.
    coord_electrode : Quantity array
        Depth of evenly spaced electrode contact points.
    sigma : Quantity float
        Conductivity of tissue.
    method : string
        CSD estimation method, either of 'standard': the standard
        double-derivative method, 'delta': delta-iCSD method, 'step':
        step-iCSD method, 'spline': spline-iCSD method. Default is 'standard'
    diam : Quantity float
        Diamater of the assumed circular planar current sources centered at
        each contact, required by iCSD methods (= 'delta', 'step',
        'spline'). Default is `None`.
    h : float or np.ndarray * quantity.Quantity
        assumed thickness of the source cylinders at all or each contact
    sigma_top : Quantity float
        Conductivity on top of tissue. When set to `None`, the same value as
        sigma: is used. Default is `None`.
    tol : float
        Tolerance of numerical integration, required by step- and
        spline-iCSD methods. Default is 1E-6.
    num_steps : int
        Number of data points for the spatially upsampled LFP/CSD data,
        required by spline-iCSD method. Default is 200.
    f_type : string
        Type of spatial
        used for smoothing of the result, either of
        'boxcar' (uses `scipy.signal.baxcar()`), 'hamming' (
        `scipy.signal.hamming()`), 'triangular' (`scipy.signal.tri()`),
        'gaussian' (`scipy.signal.gaussian`), 'identity' (no smoothing is
        applied). Default is 'identity'.
    f_order : float tuple
        Parameters to be passed to the scipy.signal function associated with
        the specified filter type.


    Returns
    -------
    tuple : (csd, csd_filtered)
        csd : neo.AnalogSignalArray
            Estimated CSD
        csd_filtered : neo.AnalogSignalArray
            Estimated CSD, spatially filtered


    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import io
    import quantities as pq
    import neo

    import icsd


    #loading test data
    test_data = io.loadmat('test_data.mat')

    #prepare lfp data for use, by changing the units to SI and append
    #quantities, along with electrode geometry and conductivities
    lfp_data = test_data['pot1'] * 1E-3 * pq.V        # [mV] -> [V]
    z_data = np.linspace(100E-6, 2300E-6, 23) * pq.m  # [m]
    diam = 500E-6 * pq.m                              # [m]
    sigma = 0.3 * pq.S / pq.m                         # [S/m] or [1/(ohm*m)]
    sigma_top = 0. * pq.S / pq.m                      # [S/m] or [1/(ohm*m)]

    lfp = neo.AnalogSignalArray(lfp_data.T, sampling_rate=2.0*pq.kHz)

    # Input dictionaries for each method
    params = {}
    params['delta'] = {
        'method': 'delta',
        'lfp' : lfp,
        'coord_electrode' : z_data,
        'diam' : diam,        # source diameter
        'sigma' : sigma,           # extracellular conductivity
        'sigma_top' : sigma,       # conductivity on top of cortex
    }
    params['step'] = {
        'method': 'step',
        'lfp' : lfp,
        'coord_electrode' : z_data,
        'diam' : diam,
        'sigma' : sigma,
        'sigma_top' : sigma,
        'tol' : 1E-12,          # Tolerance in numerical integration
        }
    params['spline'] = {
        'method': 'spline',
        'lfp' : lfp,
        'coord_electrode' : z_data,
        'diam' : diam,
        'sigma' : sigma,
        'sigma_top' : sigma,
        'num_steps' : 201,      # Spatial CSD upsampling to N steps
        'tol' : 1E-12,
        }
    params['standard'] = {
        'method': 'standard',
        'lfp' : lfp,
        'coord_electrode' : z_data,
        'sigma' : sigma,
        }

    #plot LFP signal
    fig, axes = plt.subplots(len(params)+1, 1, figsize=(6, 8))
    ax = axes[0]
    im = ax.imshow(lfp.magnitude.T, origin='upper', vmin=-abs(lfp).max(),
                   vmax=abs(lfp).max(), cmap='jet_r', interpolation='nearest')
    ax.axis(ax.axis('tight'))
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('LFP (%s)' % lfp_data.dimensionality.string)
    ax.set_xticklabels([])
    ax.set_title('LFP')
    ax.set_ylabel('ch #')
    i_ax = 1
    for method, param in params.items():
        ax = axes[i_ax]
        i_ax += 1
        csd = icsd.estimate_csd(**param)
        im = ax.imshow(csd.magnitude.T, origin='upper', vmin=-abs(csd).max(),
                       vmax=abs(csd).max(), cmap='jet_r',
                       interpolation='nearest')
        ax.axis(ax.axis('tight'))
        ax.set_title(method)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('CSD (%s)' % csd.dimensionality.string)
        ax.set_xticklabels([])
        ax.set_ylabel('ch #')

    plt.show()
    """

    supported_methods = ("standard", "delta", "step", "spline")
    icsd_methods = ("delta", "step", "spline")

    if method not in supported_methods:
        print(
            "Pamareter `method` must be either of {}".format(
                ", ".join(supported_methods)
            )
        )
        raise ValueError
    elif method in icsd_methods and diam is None:
        print(
            "Parameter `diam` must be specified for iCSD methods: {}".format(
                ", ".join(icsd_methods)
            )
        )
        raise ValueError

    if not isinstance(lfp, neo.AnalogSignalArray):
        print("Parameter `lfp` must be neo.AnalogSignalArray")
        raise TypeError

    if f_type is not "identity" and f_order is None:
        print("The order of {} filter must be specified".format(f_type))
        raise ValueError

    lfp_pqarr = lfp.magnitude.T * lfp.units
    if sigma_top is None:
        sigma_top = sigma

    arg_dict = {
        "lfp": lfp_pqarr,
        "coord_electrode": coord_electrode,
        "sigma": sigma,
        "f_type": f_type,
        "f_order": f_order,
    }
    if method == "standard":
        csd_estimator = StandardCSD(**arg_dict)
    else:
        arg_dict["diam"] = diam
        arg_dict["sigma_top"] = sigma_top
        if method == "delta":
            csd_estimator = DeltaiCSD(**arg_dict)
        else:
            arg_dict["tol"] = tol
            if method == "step":
                arg_dict["h"] = h
                csd_estimator = StepiCSD(**arg_dict)
            else:
                arg_dict["num_steps"] = num_steps
                csd_estimator = SplineiCSD(**arg_dict)
    csd_pqarr = csd_estimator.get_csd()
    csd_pqarr_filtered = csd_estimator.filter_csd(csd_pqarr)
    csd = neo.AnalogSignalArray(
        csd_pqarr.T, t_start=lfp.t_start, sampling_rate=lfp.sampling_rate
    )
    csd_filtered = neo.AnalogSignalArray(
        csd_pqarr_filtered.T, t_start=lfp.t_start, sampling_rate=lfp.sampling_rate
    )

    return csd, csd_filtered

# -*- coding: utf-8 -*-
'''icsd testing suite'''

import os
import numpy as np
import numpy.testing as nt
import quantities as pq
import scipy.integrate as si
from scipy.interpolate import interp1d
import icsd
import unittest

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


def potential_of_plane(z_j, z_i=0.*pq.m,
                      C_i=1*pq.A/pq.m**2,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of infinite horizontal plane with constant
    current source density at a vertical offset z_j. 
    
    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to source layer
    z_i : float*pq.m
        z-position of source layer
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m

    Notes
    -----
    The potential is 0 at the plane, as the potential goes to infinity for
    large distances
    
    '''
    try:
        assert(z_j.units == z_i.units)
    except AssertionError as ae:
        print('units of z_j ({}) and z_i ({}) not equal'.format(z_j.units,
                                                                    z_i.units))
        raise ae
    
    return -C_i/(2*sigma)*abs(z_j-z_i).simplified


def potential_of_disk(z_j,
                      z_i=0.*pq.m,
                      C_i=1*pq.A/pq.m**2,
                      R_i=1E-3*pq.m,
                      sigma=0.3*pq.S/pq.m):
    '''
    Return potential of circular disk in horizontal plane with constant
    current source density at a vertical offset z_j. 
    
    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z_j-position of source disk
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    '''
    try:
        assert(z_j.units == z_i.units == R_i.units)
    except AssertionError as ae:
        print('units of z_j ({}), z_i ({}) and R_i ({}) not equal'.format(
            z_j.units, z_i.units, R_i.units))
        raise ae
    
    return C_i/(2*sigma)*(np.sqrt((z_j-z_i)**2 + R_i**2) - abs(z_j-z_i)).simplified


def potential_of_cylinder(z_j,
                          z_i=0.*pq.m,
                          C_i=1*pq.A/pq.m**3,
                          R_i=1E-3*pq.m,
                          h_i=0.1*pq.m,
                          sigma=0.3*pq.S/pq.m,
                          ):
    '''
    Return potential of cylinder in horizontal plane with constant homogeneous
    current source density at a vertical offset z_j. 
    

    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z-position of center of source cylinder
    h_i : float*pq.m
        thickness of cylinder
    C_i : float*pq.A/pq.m**3
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m

    Notes
    -----
    Sympy can't deal with eq. 11 in Pettersen et al 2006, J neurosci Meth,
    so we numerically evaluate it in this function.
    
    Tested with
    
    >>>from sympy import *
    >>>C_i, z_i, h, z_j, z_j, sigma, R = symbols('C_i z_i h z z_j sigma R')
    >>>C_i*integrate(1/(2*sigma)*(sqrt((z-z_j)**2 + R**2) - abs(z-z_j)), (z, z_i-h/2, z_i+h/2))


    '''
    try:
        assert(z_j.units == z_i.units == R_i.units == h_i.units)
    except AssertionError as ae:
        print('units of z_j ({}), z_i ({}), R_i ({}) and h ({}) not equal'.format(
            z_j.units, z_i.units, R_i.units, h_i.units))
        raise ae

    #speed up tests by stripping units
    _sigma = float(sigma)
    _R_i = float(R_i)
    _z_i = float(z_i)
    _z_j = float(z_j)

    #evaluate integrand using quad
    def integrand(z):
        return 1/(2*_sigma)*(np.sqrt((z-_z_j)**2 + _R_i**2) - abs(z-_z_j))
        
    phi_j, abserr = C_i*si.quad(integrand, z_i-h_i/2, z_i+h_i/2)
    
    return (phi_j * z_i.units**2 / sigma.units)



def get_lfp_of_planes(z_j=np.arange(21)*1E-4*pq.m,
                      z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                      C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**2,
                      sigma=0.3*pq.S/pq.m,
                      plot=True):
    '''
    Compute the lfp of spatially separated planes with given current source
    density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, (zi, Ci) in enumerate(zip(z_i, C_i)):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_plane(zj, zi, Ci, sigma)
    
    #test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        for i, C in enumerate(C_i):
            ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel('z_j ({})'.format(z_j.units))
        ax.set_xlabel('C_i ({})'.format(C_i.units))
        ax.set_title('planar CSD')
    
        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel('phi_j ({})'.format(phi_j.units))
        ax.set_title('LFP')
    
    return phi_j, C_i


def get_lfp_of_disks(z_j=np.arange(21)*1E-4*pq.m,
                     z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                     C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**2,
                     R_i = np.array([1, 1, 1])*1E-3*pq.m,
                     sigma=0.3*pq.S/pq.m,
                     plot=True):
    '''
    Compute the lfp of spatially separated disks with a given
    current source density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, (zi, Ci, Ri) in enumerate(zip(z_i, C_i, R_i)):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_disk(zj, zi, Ci, Ri, sigma)
    
    #test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        for i, C in enumerate(C_i):
            ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel('z_j ({})'.format(z_j.units))
        ax.set_xlabel('C_i ({})'.format(C_i.units))
        ax.set_title('disk CSD\nR={}'.format(R_i))
    
        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel('phi_j ({})'.format(phi_j.units))
        ax.set_title('LFP')
    
    return phi_j, C_i
    

def get_lfp_of_cylinders(z_j=np.arange(21)*1E-4*pq.m,
                         z_i=np.array([8E-4, 10E-4, 12E-4])*pq.m,
                         C_i=np.array([-.5, 1., -.5])*pq.A/pq.m**3,
                         R_i = np.array([1, 1, 1])*1E-3*pq.m,
                         h_i=np.array([1, 1, 1])*1E-4*pq.m,
                         sigma=0.3*pq.S/pq.m,
                         plot=True):
    '''
    Compute the lfp of spatially separated disks with a given
    current source density
    '''
    phi_j = np.zeros(z_j.size)*pq.V
    for i, (zi, Ci, Ri, hi) in enumerate(zip(z_i, C_i, R_i, h_i)):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_cylinder(zj, zi, Ci, Ri, hi, sigma)
    
    #test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        ax.barh(np.asarray(z_i-h_i/2),
                np.asarray(C_i),
                np.asarray(h_i), color='r')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel('z_j ({})'.format(z_j.units))
        ax.set_xlabel('C_i ({})'.format(C_i.units))
        ax.set_title('cylinder CSD\nR={}'.format(R_i))
    
        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel('phi_j ({})'.format(phi_j.units))
        ax.set_title('LFP')
    
    return phi_j, C_i
    





class TestICSD(unittest.TestCase):
    '''
    Set of test functions for each CSD estimation method comparing
    estimate to LFPs calculated with known ground truth CSD
    '''
    
    def test_StandardCSD_00(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates.
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
                
        #uniform conductivity
        sigma = 0.3*pq.S/pq.m
        
        #flag for debug plots
        plot = False
        
        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_planes(z_j, z_i, C_i, sigma, plot)
        std_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'sigma' : sigma,
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)
    

    def test_StandardCSD_01(self):
        '''test using non-standard SI units 1'''
        #set some parameters for ground truth csd and csd estimates.
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*1E3*pq.A/pq.m**2
                
        #uniform conductivity
        sigma = 0.3*pq.S/pq.m
        
        #flag for debug plots
        plot = False
        
        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_planes(z_j, z_i, C_i, sigma, plot)
        std_input = {
            'lfp' : phi_j*1E3*pq.mV/pq.V,
            'coord_electrode' : z_j,
            'sigma' : sigma,
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_StandardCSD_02(self):
        '''test using non-standard SI units 2'''
        #set some parameters for ground truth csd and csd estimates.
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
                
        #uniform conductivity
        sigma = 0.3*pq.S/pq.m
        
        #flag for debug plots
        plot = False
        
        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_planes(z_j, z_i, C_i, sigma, plot)
        std_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j*1E3*pq.mm/pq.m,
            'sigma' : sigma,
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)
        
        
    def test_StandardCSD_03(self):
        '''test using non-standard SI units 3'''
        #set some parameters for ground truth csd and csd estimates.
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
                
        #uniform conductivity
        sigma = 0.3*pq.mS/pq.m
        
        #flag for debug plots
        plot = False
        
        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_planes(z_j, z_i, C_i, sigma, plot)
        std_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'sigma' : sigma*1E3*pq.mS/pq.S,
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)
        
    
    def test_DeltaiCSD_00(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(z_j, z_i, C_i, R_i, sigma,
                                      plot)
        delta_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,        # source diameter
            'sigma' : sigma,           # extracellular conductivity
            'sigma_top' : sigma_top,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }
        
        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_DeltaiCSD_01(self):
        '''test using non-standard SI units 1'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(z_j, z_i, C_i, R_i, sigma,
                                      plot)
        delta_input = {
            'lfp' : phi_j*1E3*pq.mV/pq.V,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,        # source diameter
            'sigma' : sigma,           # extracellular conductivity
            'sigma_top' : sigma_top,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }
        
        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_DeltaiCSD_02(self):
        '''test using non-standard SI units 2'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(z_j, z_i, C_i, R_i, sigma,
                                      plot)
        delta_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j*1E3*pq.mm/pq.m,
            'diam' : R_i.mean()*2*1E3*pq.mm/pq.m,        # source diameter
            'sigma' : sigma,           # extracellular conductivity
            'sigma_top' : sigma_top,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }
        
        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_DeltaiCSD_03(self):
        '''test using non-standard SI units 3'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(z_j, z_i, C_i, R_i, sigma,
                                      plot)
        delta_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,        # source diameter
            'sigma' : sigma*1E3*pq.mS/pq.S,           # extracellular conductivity
            'sigma_top' : sigma_top*1E3*pq.mS/pq.S,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }
        
        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_DeltaiCSD_04(self):
        '''test non-continous z_j array'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**2
        
        #source radius (delta, step)
        R_i = np.ones(z_j.size)*1E-3*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(z_j, z_i, C_i, R_i, sigma,
                                      plot)
        inds = np.delete(np.arange(21), 5)
        delta_input = {
            'lfp' : phi_j[inds],
            'coord_electrode' : z_j[inds],
            'diam' : R_i[inds]*2,        # source diameter
            'sigma' : sigma,           # extracellular conductivity
            'sigma_top' : sigma_top,       # conductivity on top of cortex
            'f_type' : 'gaussian',  # gaussian filter
            'f_order' : (3, 1),     # 3-point filter, sigma = 1.
        }
        
        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i[inds], csd)

    
    def test_StepiCSD_units_00(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i, C_i, R_i, h_i,
                                          sigma, plot)

        step_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'h' : h_i,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)
            

    def test_StepiCSD_01(self):
        '''test using non-standard SI units 1'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i, C_i, R_i, h_i,
                                          sigma, plot)

        step_input = {
            'lfp' : phi_j*1E3*pq.mV/pq.V,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'h' : h_i,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)

        
    def test_StepiCSD_02(self):
        '''test using non-standard SI units 2'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i, C_i, R_i, h_i,
                                          sigma, plot)

        step_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j*1E3*pq.mm/pq.m,
            'diam' : R_i.mean()*2*1E3*pq.mm/pq.m,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'h' : h_i*1E3*pq.mm/pq.m,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)


    def test_StepiCSD_03(self):
        '''test using non-standard SI units 3'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i, C_i, R_i, h_i,
                                          sigma, plot)

        step_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i.mean()*2,
            'sigma' : sigma*1E3*pq.mS/pq.S,
            'sigma_top' : sigma*1E3*pq.mS/pq.S,
            'h' : h_i,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd)

        
    def test_StepiCSD_units_04(self):
        '''test non-continous z_j array'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
        
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i, C_i, R_i, h_i,
                                          sigma, plot)
        inds = np.delete(np.arange(21), 5)
        step_input = {
            'lfp' : phi_j[inds],
            'coord_electrode' : z_j[inds],
            'diam' : R_i[inds]*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'h' : h_i[inds],
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i[inds], csd)


    def test_SplineiCSD_00(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
                
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma

        #construct interpolators, spline method assume underlying source
        #pattern generating LFPs that are cubic spline interpolates between
        #contacts so we generate CSD data relying on the same assumption 
        f_C = interp1d(z_i, C_i, kind='cubic')
        f_R = interp1d(z_i, R_i)
        num_steps = 201
        z_i_i = np.linspace(float(z_i[0]), float(z_i[-1]), num_steps)*z_i.units
        C_i_i = f_C(np.asarray(z_i_i))*C_i.units
        R_i_i = f_R(z_i_i)*R_i.units

        h_i_i = np.ones(z_i_i.size)*np.diff(z_i_i).min()
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          sigma, plot)

        spline_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'num_steps' : num_steps,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd, decimal=3)


    def test_SplineiCSD_01(self):
        '''test using standard SI units, deep electrode coordinates'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(10, 31)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
                
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma

        #construct interpolators, spline method assume underlying source
        #pattern generating LFPs that are cubic spline interpolates between
        #contacts so we generate CSD data relying on the same assumption 
        f_C = interp1d(z_i, C_i, kind='cubic')
        f_R = interp1d(z_i, R_i)
        num_steps = 201
        z_i_i = np.linspace(float(z_i[0]), float(z_i[-1]), num_steps)*z_i.units
        C_i_i = f_C(np.asarray(z_i_i))*C_i.units
        R_i_i = f_R(z_i_i)*R_i.units

        h_i_i = np.ones(z_i_i.size)*np.diff(z_i_i).min()
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          sigma, plot)

        spline_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'num_steps' : num_steps,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd, decimal=3)


    def test_SplineiCSD_02(self):
        '''test using non-standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
                
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma

        #construct interpolators, spline method assume underlying source
        #pattern generating LFPs that are cubic spline interpolates between
        #contacts so we generate CSD data relying on the same assumption 
        f_C = interp1d(z_i, C_i, kind='cubic')
        f_R = interp1d(z_i, R_i)
        num_steps = 201
        z_i_i = np.linspace(float(z_i[0]), float(z_i[-1]), num_steps)*z_i.units
        C_i_i = f_C(np.asarray(z_i_i))*C_i.units
        R_i_i = f_R(z_i_i)*R_i.units

        h_i_i = np.ones(z_i_i.size)*np.diff(z_i_i).min()
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          sigma, plot)

        spline_input = {
            'lfp' : phi_j*1E3*pq.mV/pq.V,
            'coord_electrode' : z_j,
            'diam' : R_i*2,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'num_steps' : num_steps,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd, decimal=3)


    def test_SplineiCSD_03(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
                
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma

        #construct interpolators, spline method assume underlying source
        #pattern generating LFPs that are cubic spline interpolates between
        #contacts so we generate CSD data relying on the same assumption 
        f_C = interp1d(z_i, C_i, kind='cubic')
        f_R = interp1d(z_i, R_i)
        num_steps = 201
        z_i_i = np.linspace(float(z_i[0]), float(z_i[-1]), num_steps)*z_i.units
        C_i_i = f_C(np.asarray(z_i_i))*C_i.units
        R_i_i = f_R(z_i_i)*R_i.units

        h_i_i = np.ones(z_i_i.size)*np.diff(z_i_i).min()
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          sigma, plot)

        spline_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j*1E3*pq.mm/pq.m,
            'diam' : R_i*2*1E3*pq.mm/pq.m,
            'sigma' : sigma,
            'sigma_top' : sigma,
            'num_steps' : num_steps,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd, decimal=3)
        

    def test_SplineiCSD_04(self):
        '''test using standard SI units'''
        #set some parameters for ground truth csd and csd estimates., e.g.,
        #we will use same source diameter as in ground truth
        
        #contact point coordinates
        z_j = np.arange(21)*1E-4*pq.m
        
        #source coordinates
        z_i = z_j
        
        #current source density magnitude
        C_i = np.zeros(z_i.size)*pq.A/pq.m**3
        C_i[7:12:2] += np.array([-.5, 1., -.5])*pq.A/pq.m**3
                
        #source radius (delta, step)
        R_i = np.ones(z_i.size)*1E-3*pq.m
        
        #source height (cylinder)
        h_i = np.ones(z_i.size)*1E-4*pq.m
        
        #conductivity, use same conductivity for top layer (z_j < 0)
        sigma = 0.3*pq.S/pq.m
        sigma_top = sigma

        #construct interpolators, spline method assume underlying source
        #pattern generating LFPs that are cubic spline interpolates between
        #contacts so we generate CSD data relying on the same assumption 
        f_C = interp1d(z_i, C_i, kind='cubic')
        f_R = interp1d(z_i, R_i)
        num_steps = 201
        z_i_i = np.linspace(float(z_i[0]), float(z_i[-1]), num_steps)*z_i.units
        C_i_i = f_C(np.asarray(z_i_i))*C_i.units
        R_i_i = f_R(z_i_i)*R_i.units

        h_i_i = np.ones(z_i_i.size)*np.diff(z_i_i).min()
        
        #flag for debug plots
        plot = False

        #get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          sigma, plot)

        spline_input = {
            'lfp' : phi_j,
            'coord_electrode' : z_j,
            'diam' : R_i*2,
            'sigma' : sigma*1E3*pq.mS/pq.S,
            'sigma_top' : sigma*1E3*pq.mS/pq.S,
            'num_steps' : num_steps,
            'tol' : 1E-12,          # Tolerance in numerical integration
            'f_type' : 'gaussian',
            'f_order' : (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()
        
        self.assertEqual(C_i.units, csd.units)
        nt.assert_array_almost_equal(C_i, csd, decimal=3)

                
#def suite(verbosity=2):
#    '''
#    Run unittests for the CSD toolbox
#    
#    
#    Arguments
#    ---------
#    verbosity : int
#        verbosity level
#        
#    '''
#    suite = unittest.TestLoader().loadTestsFromTestCase(TestICSD)
#    unittest.TextTestRunner(verbosity=verbosity).run(suite)
#    
#
#
#if __name__ == '__main__':
#    suite()


def suite():
    suite = unittest.makeSuite(TestICSD, 'test')
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

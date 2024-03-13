import os
import math
import pandas as pd
import numpy as np
import six
from neuron import h

from scipy.interpolate import NearestNDInterpolator as NNip
from scipy.interpolate import LinearNDInterpolator as Lip

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.modules.xstim_waveforms import stimx_waveform_factory

class ComsolMod(SimulatorMod):
    """ 
    __init__: COMSOL output .txt file is loaded as pandas dataframe and then used to set up nearest neighbour (NN) interpolation object to create interpolation map
    :param comsol_file: path of .txt file. Coordinates in [um], potentials in [mV], timepoints in [s].
    :param waveform: path of .csv file as created with examples/bio_components/waveform.py
                    If specified, comsol_file should contain output from stationary study and waveform is defined through this parameter.
                    If not specified, comsol_file should contain output from time-dependent study.

    initialise: An interpolation map is defined of every segment and stored in dictionary self._NN, done iteratively for every cell/gid.
    The interpolation map points (the center of) every segment to its NN. It is calculated once here and then used in every step. 
    Next, the COMSOL output is also interpolated in time to match the timesteps in BMTK.

    step: The interpolation map is used to point each segment to its NN and find the corresponding voltage value in the comsol df.
    """

    def __init__(self, comsol_file, waveform=None, cells=None, set_nrn_mechanisms=True,
                 node_set=None, amplitude=1, ip_method='NN'):

        if waveform is not None:
            self._waveform = stimx_waveform_factory(waveform)
        else:
            self._waveform = None
        
        self._amplitude = amplitude
        self._ip_method = ip_method
        self._comsol_file = comsol_file

        # extract useful information from header row in COMSOL output .txt file. 
        header = pd.read_csv(comsol_file, sep="\s{3,}", header=None, skiprows=8, nrows=1, engine='python').to_numpy()[0] # load header row of .txt file
        header[0] = header[0][2:]                               # remove '% ' before first column name
        if header[3][3] == 'V':
            self._unit = 1000
        elif header[3][3] == 'm':
            self._unit = 1
        for i,col in enumerate(header):                         # remove superfluous characters before actual time value
            if col[0] == "V":
                if self._waveform is None:
                    header[i] = float(col[11:])
                else:
                    header[i] = 0
        self._timepoints = np.array(header[3:], dtype=float)    # create array of timepoints  

        # load data in COMSOL output .txt file.  
        self._comsol = pd.read_csv(comsol_file, sep="\s+", header=None, skiprows=9, names=header)           # load data from .txt file
        
        if self._ip_method == 'NN':
            self._NNip = NNip(self._comsol[['x','y','z']], np.arange(len(self._comsol['x'])))               # create scipy NN interpolation object 
            self._NN = {}                                           # initialise empty dictionary that will contain NN map of each cell
        
        elif self._ip_method == 'L':                 # Only works if waveform is specified
            self._Lip = Lip(self._comsol[['x','y','z']], self._comsol[0])
            self._L = {}

        self._set_nrn_mechanisms = set_nrn_mechanisms
        self._cells = cells
        self._local_gids = []
        

    def initialize(self, sim):
        if self._cells is None:
            # if specific gids not listed just get all biophysically detailed cells on this rank
            self._local_gids = sim.biophysical_gids
        else:
            # get subset of selected gids only on this rank
            self._local_gids = list(set(sim.local_gids) & set(self._all_gids))

        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            cell.setup_xstim(self._set_nrn_mechanisms)

            # spatial interpolation
            r05 = cell.seg_coords.p05               # get position of middle of segment
            if self._ip_method == 'NN':
                self._NN[gid] = self._NNip(r05.T)       # get nearest COMSOL node
            elif self._ip_method == 'L':
                self._L[gid] = self._Lip(r05.T)

        if self._waveform is None:
            # temporal interpolation
            dt = sim.dt/1000                                                            # BMTK uses [ms], COMSOL uses [s]
            tsteps = np.arange(self._timepoints[0], self._timepoints[-1]+dt, dt)        # interpolate time axis
            self._arr = np.zeros((self._comsol.shape[0],len(tsteps)))                   # initialise empty array
            for i in range(self._comsol.shape[0]):                                      # update each row (corresponding to a COMSOL node) of the array with the time-interpolated values
                self._arr[i,:] = np.interp(tsteps, self._timepoints, self._comsol.iloc[i,3:]).flatten()
        
        else:
            self._arr = self._comsol[0].to_numpy().flatten()

    def step(self, sim, tstep):
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)        # get cell gid
            if self._ip_method == 'NN':
                NN = self._NN[gid]                      # vector that points each node of the cell to its nearest neighbour in the .txt file
                if self._waveform is None:
                    T = int(1000*self._timepoints[-1]/sim.dt)
                    tstep = tstep % T                   # In case of periodic stimulation
                    v_ext = self._arr[NN,tstep+1]       # assign extracellular potential value of NN at tstep
                else:
                    T = int(self._waveform.definition["time"].iloc[-1])
                    tstep = tstep % T                   # In case of periodic stimulation
                    v_ext = self._arr[NN]*self._waveform.calculate(tstep+1)
            elif self._ip_method == 'L':
                L = self._L[gid]
                T = int(self._waveform.definition["time"].iloc[-1])
                tstep = tstep % T                       # In case of periodic stimulation
                v_ext = L*self._waveform.calculate(tstep+1)

            v_ext *= self._amplitude * self._unit

            cell.set_e_extracellular(h.Vector(v_ext))

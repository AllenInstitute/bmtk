import pandas as pd
import numpy as np
from neuron import h

from scipy.interpolate import NearestNDInterpolator as NNip
from scipy.interpolate import LinearNDInterpolator as Lip

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.modules.xstim_waveforms import stimx_waveform_factory

class ComsolMod(SimulatorMod):
    """This module takes extracellular potentials that were calculated in COMSOL and imposes them on a biophysically detailed network. 
    It is similar to BioNet's xstim module, but offers the additional flexibility of FEM, instead of relying on simplified analytical solutions.
    As such, this module allows to use BMTK as a part of the hybrid modelling approach, 
    where extracellular potentials are calculated using FEM in a first step and then imposed on a model of a neuronal network in a second step.
    """

    def __init__(self, comsol_files, waveforms=None, amplitudes=1, 
                 cells=None, set_nrn_mechanisms=True, node_set=None):
        """Checks if a waveform argument was passed which determines what comsol_files and amplitudes should look like.
        If no waveform is specified:
            The comsol output should be from a time-dependent study.
            The amplitude can optionally be passed in the form of an integer to scale all potentials.
        If one or more waveforms are specified:
            There should be as many comsol outputs from stationary studies. 
            Optionally, as many amplitudes can be passed to scale the corresponding potentials

        :param comsol_files: (str or list) "/path/to/comsol.txt" or list thereof.
        :param waveforms: (str or list) "/path/to/waveform.csv" or list thereof. Defaults to None, in which case comsol study should be time dependent.
        :param amplitudes: (int or list) waveform amplitude or list thereof. Defaults to 1.
        :param cells: defaults to None.
        :param set_nrn_mechanisms: defaults to True.
        :param node_set: defaults to None.
        """

        if waveforms is None:
            self._comsol_files = comsol_files 
            self._waveforms = waveforms
            self._amplitudes = amplitudes

        else:
            self._comsol_files = comsol_files if type(comsol_files) is list else [comsol_files]
            self._nb_files = len(self._comsol_files) 
            self._waveforms = waveforms if type(waveforms) is list else [waveforms]
            amplitudes = amplitudes if type(amplitudes) is list else [amplitudes]
            self._amplitudes = amplitudes*len(self._comsol_files) if len(amplitudes) == 1 else amplitudes
            
            try:
                assert self._nb_files == len(self._comsol_files) == len(self._waveforms) == len(self._amplitudes)
            except AssertionError:
                print("AssertionError: comsol_files, waveforms, and amplitudes have a different length.")

            self._data = [None]*self._nb_files

        self._set_nrn_mechanisms = set_nrn_mechanisms
        self._cells = cells
        self._local_gids = []
        

    def initialize(self, sim):
        """Checks if a waveform argument was passed which determines how to comsol.txt and waveform.csv should be treated.
        If no waveform is specified:
            Loads COMSOL output
            Sets up nearest neighbour interpolation object (for spatial interpolation)
            Performs temporal interpolation so COMSOL and BMTK timings match.
        If one or more waveforms are specified:
            Iterates over COMSOL outputs:
                Loads COMSOL output
                Iterates over cells:
                    Retrieves potentials at each segment via spatial interpolation

        :param sim: Simulation object
        """        
        if self._cells is None:
            # if specific gids not listed just get all biophysically detailed cells on this rank
            self._local_gids = sim.biophysical_gids
        else:
            # get subset of selected gids only on this rank
            self._local_gids = list(set(sim.local_gids) & set(self._all_gids))
        
        if self._waveforms is None: # If time-dependent COMSOL study
            self._data =  self.load_comsol(self._comsol_files)      # Load COMSOL file
            # Set up interpolator that points every cell segment to the closest COMSOL node
            self._NNip = NNip(self._data[['x','y','z']], np.arange(len(self._data['x'])))
            self._NN = {}
            
            for gid in self._local_gids: # Iterate over cells
                cell = sim.net.get_cell_gid(gid)
                cell.setup_xstim(self._set_nrn_mechanisms)
                
                r05 = cell.seg_coords.p05               # Get position of segment centre
                self._NN[gid] = self._NNip(r05.T)       # Create map that points each segment to the closest COMSOL node
                
            # Temporal interpolation
            timestamps_comsol = np.array(list(self._data)[3:], dtype=float)[:]                          # Retrieve array of timestamps in COMSOL
            timestamps_bmtk = np.arange(timestamps_comsol[0], timestamps_comsol[-1]+sim.dt, sim.dt)     # Create array of timestamps in BMTK
            self._data_temp = np.zeros((self._data.shape[0], len(timestamps_bmtk)))                     # Start with empty array
            for i in range(self._data.shape[0]):                                 
                self._data_temp[i,:] = np.interp(timestamps_bmtk, timestamps_comsol, self._data.iloc[i,3:]).flatten()                                                          
            self._data = self._data_temp*self._amplitudes
            max_time = timestamps_bmtk[-1]    
            self._period = int(max_time/sim.dt)

        else: # Else stationary study    
            self._Lip = [None]*self._nb_files
            self._L = [None]*self._nb_files

            for i in range(self._nb_files):     # For each COMSOL file                                           
                self._data[i] =  self.load_comsol(self._comsol_files[i])            # Load COMSOL file
                self._waveforms[i] = stimx_waveform_factory(self._waveforms[i])     # Load waveform                                                  

                self._Lip[i] = Lip(self._data[i][['x','y','z']], self._data[i][0])  # Create interpolator
                self._L[i] = {}                                                         

            for gid in self._local_gids:        # Iterate over cells                          
                cell = sim.net.get_cell_gid(gid)
                cell.setup_xstim(self._set_nrn_mechanisms)

                r05 = cell.seg_coords.p05                       # Get position of middle of segment
                for i in range(self._nb_files):                 # For every COMSOL file
                    self._L[i][gid] = self._Lip[i](r05.T)       # Retrieve potentials with interpolate

                
    def step(self, sim, tstep):
        """Checks if a waveform argument was passed which determines how potentials should be retrieved.
        Iterates over all cells:
            If no waveform is specified:
                Retrieves nearest neighbour with interpolator
                Looks up potentials (for each segment of the cell) in comsol data at current time
            If one or more waveforms are specified:
                Initialises extracellular potential v_ext at 0
                Iterates over COMSOL outputs (thus making a linear combination of several FEM solutions):
                    Calls interpolated potentials
                    Multiplies by corresponding waveform value at current time and by corresponding amplitude
                    Adds to v_ext
                    
        :param sim: Simulation object
        :param tstep: (int) timestep
        """
        
        for gid in self._local_gids:        # Iterate over cells                                            
            cell = sim.net.get_cell_gid(gid)                                    

            
            if self._waveforms is None:             # If time-dependent COMSOL study                                        
                NN = self._NN[gid]                  # Point each node of the cell to the nearest COMSOL node
                tstep = tstep % self._period        # Repeat periodic stimulation
                v_ext = self._data[NN, tstep]       # Look up extracellular potentials at current time
             
            else:       # Else stationary study          
                v_ext = np.zeros(np.shape(self._L[0][gid]))     # Initialise v_ext as zero array               
               
                for i in range(self._nb_files):     # Iterate over COMSOL studies                                 
                    period = self._waveforms[i].definition["time"].iloc[-1]     # Get duration of waveform.csv
                    simulation_time = (tstep + 1) * sim.dt                      # Calculate current time in simulation run
                    simulation_time = simulation_time % period                  # Repeat periodic stimulation
                    # Add potentials(x,y,z)*waveform(t)*amplitude of this iteration to v_ext
                    v_ext += self._L[i][gid]*self._waveforms[i].calculate(simulation_time)*self._amplitudes[i]
                v_ext[np.isnan(v_ext)] = 0
            # if tstep == 10 and gid == 10:
            #     print(v_ext)
            cell.set_e_extracellular(h.Vector(v_ext))       # Set extracellular potentials to v_ext 

    def load_comsol(self, comsol_file):
        """Extracts data and headers from comsol.txt. Returns pandas DataFrame.
        The first three columns are the x-, y-, and z-coordinates of the solution nodes.
        For a stationary comsol study, the potentials are stored in the fourth column.
        For a time-dependent study, each subsequent column stores the potentials at one timepoints.

        :param comsol_file: (str) "/path/to/comsol.txt"
        :return: (pd DataFrame) Potentials extracted from comsol.txt
        """

        # Extract column headers and data from comsol_file
        headers = pd.read_csv(comsol_file, sep="\s{3,}", header=None, skiprows=8, nrows=1, engine='python')
        headers = headers.to_numpy()[0]
        data = pd.read_csv(comsol_file, sep="\s+", header=None, skiprows=9)

        # Convert V to mV if necessary
        if headers[3][3] == 'V':                        
            data.iloc[:,3:] *= 1000                     

        # Extract useful info from headers
        headers[0] = headers[0][2:]                     # Remove '% ' before first column name
        for i,col in enumerate(headers[3:]):            # Iterate over all elements in the header except first 3
            if len(data.columns) > 4:                   # If time-dependent comsol study
                for j, c in enumerate(col):
                    if c.isdigit():
                        break
                headers[i+3] = 1000*float(col[j:])      # Remove superfluous characters and convert from s to ms
            else:                                       # Else stationary study
                headers[i+3] = 0                        # Rename 4th column
        
        # Rename data with correct column headers
        data.columns = headers

        return data



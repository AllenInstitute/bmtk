# coding=utf-8
### Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class CreateWaveform:
    """
    Class to create a current waveform and write it to a .csv file.
    This file can be used for BioNet's xstim and comsol modules.
    To provide the comsol module with a periodic pulse train, it suffices to generate a single period of the pulse train.
    The comsol module will then automatically repeat the waveform for the duration of the simulation.
    """

    def __init__(self, piecewise, max=None, dt=0.025, path=None, plot=False):
        """ Main function when calling CreateWaveform(). It 

        :param piecewise: piecewise description of waveform. Each row defines a piece [t_stop, lambda(t)]. t_start of the piece is 0 or t_stop of the previous piece, the lambda expression defines the function as a function of time t. 
        :type piecewise: ndarray
        :param amplitude: If specified, normalise waveform to [-amplitude, amplitude]. Defaults to None.
        :type amplitude: int or None, optional
        :param dt: timestep in ms, defaults to 0.025
        :type dt: float, optional
        :param path: if not None, path to the file where the waveform values are saved. Defaults to None.
        :type path: str or None, optional
        :param plot: if true, the waveform shape is plotted. Defaults to False.
        :type plot: bool, optional
        """
    	
        self.piecewise = piecewise
        self.max = max
        self.dt = dt
        self.path = path
        self.plot = plot
            
        self.create_waveform()

        self.normalise() if self.max is not None else None

        self.write_to_csv() if self.path is not None else None

        self.plot_waveform() if self.plot is True else None

        return

    def create_waveform(self):
        """ Store list of timepoints in self.times and calculates corresponding amplitudes, storing them in self.amplitudes. """ 

        self.t_start = 0
        self.times = []
        self.amplitudes = []

        for piece in self.piecewise:
            self.t_stop, self.func = piece
            for t in np.arange(self.t_start, self.t_stop, self.dt):
                self.times.append(t)
                self.amplitudes.append(self.func(t))
            self.t_start = self.t_stop

        self.times = np.round(self.times,10)

    def normalise(self, max=None):
        """ Scale the waveform such that the absolute value of the largest peak is equal to self.max.

        :param max: Maximum value of the largest peak (in absolute terms). Defaults to None.
        :type max: int
        """          

        self.max = max if max is not None else self.max
        assert self.max is not None

        self.amplitudes = self.amplitudes/np.max(np.abs(self.amplitudes))*self.max

    def write_to_csv(self, path=None):
        """ Write self.times and self.amplitudes to the .csv specified in self.path.

        :param path: path to waveform.csv. defaults to None.
        :type path: path
        """          
        
        self.path = path if path is not None else self.path
        #assert os.path.exists(os.path.dirname(self.path))
        
        df = pd.DataFrame({'time': self.times, 'amplitude': self.amplitudes})
        df.to_csv(self.path, sep='\t', index=False)
        return

    def plot_waveform(self):
        """ Plot the waveform stored in self.times and self.amplitudes. """             

        plt.plot(np.append(0,self.times), np.append(0,self.amplitudes)) # Have the line start in (0,0) for visualisation purposes
        plt.xlabel('time [ms]')                         
        plt.ylabel('amplitude [µA]')
        plt.tight_layout()
        plt.show()

        return


def CreateBlockWaveform(n_pulses, phase_1_expr, amp_1_expr, T_1_expr, phase_2_expr, amp_2_expr, T_2_expr, save_name=None):
    """Create a block waveform using the CreateWaveform class. Except for n_pulses, all arguments should be lambda expressions of n (units: ms).
    Example:: Constant phase_1 (0.1ms): phase_1_expr = lambda n:0.1
    Example:: For phase_1 that starts at 0.1 ms and gets 0.01 ms longer after each pulse: phase_1_expr = lambda n:0.1+n/10

    :param n_pulses: number of pulses that the waveform will be made up of 
    :type n_pulses: int
    :param phase_1_expr: duration of first phase of pulse in ms
    :param phase_1_expr: duration of first phase of pulse in ms
    :type phase_1_expr: lambda
    :param amp_1_expr: amplitude of first phase of pulse in µA
    :type amp_1_expr: lambda
    :param T_1_expr: time between end of first phase and start of second phase in ms
    :type T_1_expr: lambda
    :param phase_2_expr: duration of second phase of pulse in ms
    :type phase_2_expr: lambda
    :param amp_2_expr: amplitude of second phase of pulse in µA
    :type amp_2_expr: lambda
    :param T_2_expr: time between end of one pulse and start of next pulse in ms
    :type T_2_expr: lambda
    :return: A piecewise description of the waveform that can be passed to CreateWaveform(). (2 x 4*n_pulses) array whose rows look like [t_stop, lambda].
    :rtype: ndarray
    """

    # Initialisation
    piecewise = np.zeros((4*n_pulses,2))
    t_start = 0 

    for i in range(n_pulses):
        
        # Get pulse parameters for pulse i 
        phase_1 = phase_1_expr(i)
        amp_1 =  amp_1_expr(i)
        T_1 =  T_1_expr(i)
        phase_2 = phase_2_expr(i)
        amp_2 = amp_2_expr(i)
        T_2 = T_2_expr(i)

        # Construct piecewise definition of pulse i
        piecewise_temp1 = [t_start+phase_1, lambda t, amp_1=amp_1:amp_1]
        piecewise_temp2 = [t_start+phase_1+T_1, lambda t:0]
        piecewise_temp3 = [t_start+phase_1+T_1+phase_2, lambda t, amp_2=amp_2:amp_2]
        piecewise_temp4 = [t_start+phase_1+T_1+phase_2+T_2, lambda t:0]
        piecewise = np.vstack((piecewise, piecewise_temp1, piecewise_temp2, piecewise_temp3, piecewise_temp4))  # Add pulse i
        
        t_start = t_start+phase_1+T_1+phase_2+T_2   # Update t_start

    # Construct path and pass piecewise to CreateWaveform() 
    dir_path = os.path.dirname(os.path.realpath(__file__))                                  # Directory of this file: waveform.py           
    path = os.path.join(dir_path, save_name) if save_name is not None else None      # Save .csv file in /.../stimulations/
    CreateWaveform(piecewise, max=1, path=path, plot=True)

    return piecewise


if __name__ == '__main__':
    '''
    If you run the file waveform.py instead of calling if from another file, this part will run.
    
    '''
    CreateBlockWaveform(
        n_pulses = 1,
        phase_1_expr = lambda n:0.2,
        amp_1_expr = lambda n:-4,
        T_1_expr = lambda n:0,
        phase_2_expr = lambda n:0.8,
        amp_2_expr = lambda n:1,
        T_2_expr = lambda n:4,
        save_name = "../bio_components/stimulations/waveform.csv"
    )
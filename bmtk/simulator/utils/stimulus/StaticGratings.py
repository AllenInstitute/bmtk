import numpy as np
import pandas as pd


class StaticGratings (object):

    def __init__(self,orientations=30.0*np.arange(6),spatial_frequencies=0.01*(2.0**np.arange(1,6)),phases=0.25*np.arange(4),num_trials=50, start_time=0, trial_length=250):

        self.orientations = orientations
        self.spatial_frequencies = spatial_frequencies
        self.phases = phases
        self.num_trials = num_trials
        self.start_time = start_time
        self.trial_length = trial_length

        trial_stims = np.array([ [orientation, spat_freq, phase] for orientation in self.orientations for spat_freq in self.spatial_frequencies for phase in self.phases ])

        trial_stims = np.tile(trial_stims,(num_trials,1))

        indices = np.random.permutation(trial_stims.shape[0])
        trial_stims = trial_stims[indices]

        self.stim_table = pd.DataFrame(trial_stims,columns=['orientation','spatial_frequency','phase'])

        T = self.stim_table.shape[0]
        self.T = T
        start_time_array = trial_length*np.arange(self.T) + start_time
        end_time_array = start_time_array + trial_length

        self.stim_table['start'] = start_time_array
        self.stim_table['end'] = end_time_array

    def get_image_input(self,new_size=(64,112),pix_per_degree=1.0, dtype=np.float32, add_channels=False):

        y, x = new_size
        stim_template = np.empty([self.T, y, x],dtype=dtype)

        for t, row in self.stim_table.iterrows():
            ori, sf, ph = row[0], row[1], row[2]

            theta = ori*np.pi/180.0 #convert to radians

            k = (sf/pix_per_degree)  # radians per pixel
            ph = ph*np.pi*2.0

            X,Y = np.meshgrid(np.arange(x),np.arange(y))
            X = X - x/2
            Y = Y - y/2
            Xp, Yp = self.rotate(X,Y,theta)

            stim_template[t] = np.cos(2.0*np.pi*Xp*k + ph)

        self.stim_template = stim_template

        if add_channels:
            return stim_template[:,:,:,np.newaxis]
        else:
            return stim_template

    @staticmethod
    def rotate(X,Y, theta):

        Xp = X*np.cos(theta) - Y*np.sin(theta)
        Yp = X*np.sin(theta) + Y*np.cos(theta)

        return Xp, Yp

    @classmethod
    def with_brain_observatory_stimulus(cls, num_trials=50):

        orientations = 30.0*np.arange(6)
        spatial_frequencies = 0.01*(2.0**np.arange(1,6))
        phases = 0.25*np.arange(4)

        start_time = 0
        trial_length = 250

        return cls(orientations=orientations,spatial_frequencies=spatial_frequencies,phases=phases,num_trials=num_trials,start_time=start_time,trial_length=trial_length)

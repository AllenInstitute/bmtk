import matplotlib.pyplot as plt
import numpy as np
from utilities import convert_tmin_tmax_framerate_to_trange


class Movie(object):
    def __init__(self, data, row_range=None, col_range=None, labels=('time', 'y', 'x'),
                 units=('second', 'pixel', 'pixel'), frame_rate=None, t_range=None):
        self.data = data
        self.labels = labels
        self.units = units
        assert units[0] == 'second'
        
        if t_range is None:
            self.frame_rate = float(frame_rate)
            self.t_range = np.arange(data.shape[0])*(1./self.frame_rate)
        else:
            self.t_range = np.array(t_range)
            self.frame_rate = 1./np.mean(np.diff(t_range))
            
        if row_range is None:
            self.row_range = np.arange(data.shape[1])
        else:
            self.row_range = np.array(row_range)
        if col_range is None:
            self.col_range = np.arange(data.shape[2])
        else:
            self.col_range = np.array(col_range)

    def imshow_summary(self, ax=None, show=True, xlabel=None):
        if ax is None:
            _, ax = plt.subplots(1,1)
        
        t_vals = self.t_range.copy()
        y_vals = self.data.mean(axis=2).mean(axis=1)
        ax.plot(t_vals, y_vals)
        ax.set_ylim(y_vals.min()-np.abs(y_vals.min())*.05, y_vals.max()+np.abs(y_vals.max())*.05)
        
        if not xlabel is None:
            ax.set_xlabel(xlabel)
            
        ax.set_ylabel('Average frame intensity')
        
        if show == True:
            plt.show()
            
        return ax, (t_vals, y_vals)
    
    def imshow(self, t, show=True, vmin=-1, vmax=1, cmap=plt.cm.gray):
        ti = int(t*self.frame_rate)
        data = self.data[ti,:,:]
        plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        if show:
            plt.show()

    def __add__(self, other):
        
        assert self.labels == other.labels
        assert self.units == other.units
        assert self.frame_rate == other.frame_rate
        np.testing.assert_almost_equal(self.col_range, other.col_range)
        np.testing.assert_almost_equal(self.row_range, other.row_range)
        

        new_data = np.empty((len(self.t_range)+len(other.t_range)-1, len(self.row_range), len(self.col_range)))
        new_data[:len(self.t_range), :,:] = self.data[:,:,:]
        new_data[len(self.t_range):, :,:] = other.data[1:,:,:]
        
        return Movie(new_data, row_range=self.row_range.copy(), col_range=self.col_range.copy(), labels=self.labels, units=self.units, frame_rate=self.frame_rate)
        
    @property
    def ranges(self):
        return self.t_range, self.row_range, self.col_range
    
    def get_nwb_GrayScaleMovie(self):

        t_scale = nwb.Scale(self.t_range, 'time', self.units[0])
        row_scale = nwb.Scale(self.row_range, 'distance', self.units[1])
        col_scale = nwb.Scale(self.col_range, 'distance', self.units[2])

        return nwb.GrayScaleMovie(self.data, scale=(t_scale, row_scale, col_scale))
    
    def __getitem__(self, *args):
        return self.data.__getitem__(*args)


class FullFieldMovie(Movie):
    def __init__(self, f, row_range, col_range, frame_rate=24):
        self.row_range = row_range
        self.col_range = col_range
        self.frame_size = (len(self.row_range), len(self.col_range))
        self._frame_rate = frame_rate
        self.f = f
        
    @property
    def frame_rate(self):
        return self._frame_rate
    
    @property
    def data(self):
        return self
    
    def __getitem__(self, *args):
        
        t_inds, x_inds, y_inds = args[0]
        
        assert (len(x_inds) == len(y_inds)) and (len(y_inds) == len(t_inds))
        
        # Convert frame indices to times:
        t_vals = (1./self.frame_rate)*t_inds
        
        # Evaluate and return:
        return self.f(t_vals)
    
    def full(self, t_min=0, t_max=None):
        # Compute t_range
        t_range = convert_tmin_tmax_framerate_to_trange(t_min, t_max, self.frame_rate)
         
        nt = len(t_range)
        nr = len(self.row_range)
        nc = len(self.col_range)
        a,b,c = np.meshgrid(range(nt),range(nr),range(nc))
        af, bf, cf = map(lambda x: x.flatten(), [a,b,c])
        data = np.empty((nt, nr, nc))
        data[af, bf, cf] = self.f(t_range[af])
        
        return Movie(data, row_range=self.row_range, col_range=self.col_range, labels=('time', 'y', 'x'), units=('second', 'pixel', 'pixel'), frame_rate=self.frame_rate)


class FullFieldFlashMovie(FullFieldMovie):
    def __init__(self, row_range, col_range,  t_on, t_off, max_intensity=1, frame_rate=24):
        assert t_on < t_off

        def f(t):
            return np.piecewise(t, *zip(*[(t < t_on, 0), (np.logical_and(t_on <= t, t < t_off), max_intensity),
                                          (t_off <= t, 0)]))

        super(FullFieldFlashMovie, self).__init__(f, row_range, col_range,  frame_rate=frame_rate)


class GratingMovie(Movie):
    def __init__(self, row_size, col_size, frame_rate=1000.):
        self.row_size = row_size                        #in degrees
        self.col_size = col_size                        #in degrees
        self.frame_rate = float(frame_rate)             #in Hz

    def create_movie(self, t_min = 0, t_max = 1, gray_screen_dur = 0, cpd = 0.05, temporal_f = 4, theta = 45, phase = 0., contrast = 1.0, row_size_new = None, col_size_new = None):
        """Create the grating movie with the desired parameters
        :param t_min: start time in seconds
        :param t_max: end time in seconds
        :param gray_screen_dur: Duration of gray screen before grating stimulus starts
        :param cpd: cycles per degree
        :param temporal_f: in Hz
        :param theta: orientation angle
        :return: Movie object of grating with desired parameters
        """
        assert contrast <= 1, "Contrast must be <= 1"
        assert contrast > 0, "Contrast must be > 0"

        physical_spacing = 1. / (float(cpd) * 10)    #To make sure no aliasing occurs
        self.row_range = np.linspace(0, self.row_size, self.row_size / physical_spacing, endpoint = True)
        self.col_range = np.linspace(0, self.col_size, self.col_size / physical_spacing, endpoint = True)
        numberFramesNeeded = int(round(self.frame_rate * (t_max - gray_screen_dur))) + 1
        time_range = np.linspace(gray_screen_dur, t_max - gray_screen_dur, numberFramesNeeded, endpoint=True)

        tt, yy, xx = np.meshgrid(time_range, self.row_range, self.col_range, indexing='ij')

        thetaRad = -np.pi*(180-theta)/180.
        phaseRad = np.pi*(180-phase)/180.
        xy = xx * np.cos(thetaRad) + yy * np.sin(thetaRad)
        data = contrast*np.sin(2*np.pi*(cpd * xy + temporal_f *tt) + phaseRad)

        if row_size_new != None:
            self.row_range = np.linspace(0, row_size_new, data.shape[1], endpoint = True)
        if col_size_new != None:
            self.col_range = np.linspace(0, col_size_new, data.shape[2], endpoint = True)

        if gray_screen_dur > 0:
            # just adding one or two seconds to gray screen so flash never "happens"
            m_gray = FullFieldFlashMovie(self.row_range, self.col_range, gray_screen_dur + 1, gray_screen_dur + 2,
                                         frame_rate=self.frame_rate).full(t_max=gray_screen_dur)
            mov = m_gray + Movie(data, row_range=self.row_range, col_range=self.col_range, labels=('time', 'y', 'x'),
                                 units=('second', 'pixel', 'pixel'), frame_rate=self.frame_rate)
        else:
            mov = Movie(data, row_range=self.row_range, col_range=self.col_range, labels=('time', 'y', 'x'),
                        units=('second', 'pixel', 'pixel'), frame_rate=self.frame_rate)

        return mov


if __name__ == "__main__":
    m1 = FullFieldFlashMovie(range(60), range(80), 1, 2).full(t_max=2)
    m2 = FullFieldFlashMovie(range(60), range(80), 1, 2).full(t_max=2)
    m3 = m1+m2
    m3.imshow_summary()

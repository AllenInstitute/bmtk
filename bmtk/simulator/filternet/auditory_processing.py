import pycochleagram.cochleagram as cgram
import pycochleagram.erbfilter as erb
from pycochleagram import utils
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly


# Audio file in the format of *.wav

class AuditoryInput(object):
    def __init__(self, aud_fn, low_lim=50.0, hi_lim=8000.0, sample_factor=4, downsample=None):
        """Preprocesses wave files into cochleagrams - note these are based on human auditory filters

        :param aud_fn: string, Input audio filename in the format of *.wav
        :param low_lim: float, low end of frequency range (Hz)
        :param hi_lim: float, high end of frequency range (Hz)
        :param sample_factor: int,
        """
        self.stim_array, self.sr = utils.wav_to_array(aud_fn)
        self.sample_factor = sample_factor  # density of sampling, can be 1,2, or 4
        self.low_lim = low_lim
        self.hi_lim = hi_lim
        self.downsample = downsample

    def get_cochleagram(self, desired_sr=1000, interp_to_freq=False):
        n = int(np.floor(erb.freq2erb(self.hi_lim) - erb.freq2erb(self.low_lim)) - 1)
        human_coch = cgram.human_cochleagram(self.stim_array, self.sr, n=n, sample_factor=self.sample_factor,
                                             downsample=self.downsample, nonlinearity='power', strict=False)

        filts, center_freqs, freqs = erb.make_erb_cos_filters_nx(self.stim_array.shape[0],
                                                                 self.sr, n, self.low_lim, self.hi_lim,
                                                                 self.sample_factor, padding_size=None,
                                                                 full_filter=True, strict=False)
        if interp_to_freq:
            log_freqs = np.geomspace(np.min(center_freqs), np.max(center_freqs), len(center_freqs))

            n_t = human_coch.shape[1]
            Ytf = np.empty((len(log_freqs), n_t))
            for i in range(n_t):
                f = interp1d(np.log2(center_freqs), human_coch[:, i], kind='cubic')
                Ytf[:, i] = f(np.log2(log_freqs))
            human_coch = Ytf
            center_freqs = log_freqs

        inds_keep = np.argwhere((center_freqs >= self.low_lim) & (center_freqs <= self.hi_lim))
        center_freqs = center_freqs[inds_keep]
        human_coch = human_coch[np.squeeze(inds_keep)]
        center_freqs_log = np.log2(center_freqs/np.min(center_freqs))
        human_coch = resample_poly(human_coch, desired_sr, self.sr, axis=1)
        times = np.linspace(0, 1/desired_sr * (human_coch.shape[1]-1), human_coch.shape[1])


        return human_coch, center_freqs_log, times
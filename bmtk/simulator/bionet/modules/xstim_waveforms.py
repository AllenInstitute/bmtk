import os
import numpy as np
import pandas as pd
import json
from six import string_types

from bmtk.simulator.bionet.io_tools import io

class BaseWaveform(object):
    """Abstraction of waveform class to ensure calculate method is implemented"""
    def calculate(self, simulation_time):
        raise NotImplementedError("Implement specific waveform calculation")


class BaseWaveformType(object):
    """Specific waveform type"""
    def __init__(self, waveform_config):
        self.amp = float(waveform_config["amp"]) # units? mA?
        self.delay = float(waveform_config["del"]) # ms
        self.duration = float(waveform_config["dur"]) # ms

    def is_active(self, simulation_time):
        stop_time = self.delay + self.duration
        return self.delay < simulation_time < stop_time


class WaveformTypeDC(BaseWaveformType, BaseWaveform):
    """DC (step) waveform"""
    def __init__(self, waveform_config):
        super(WaveformTypeDC, self).__init__(waveform_config)

    def calculate(self, t): # TODO better name
        if self.is_active(t):
            return self.amp
        else:
            return 0


class WaveformTypeSin(BaseWaveformType, BaseWaveform):
    """Sinusoidal waveform"""
    def __init__(self, waveform_config):
        super(WaveformTypeSin, self).__init__(waveform_config)
        self.freq = float(waveform_config["freq"])  # Hz
        self.phase_offset = float(waveform_config.get("phase", np.pi))  # radians, optional
        self.amp_offset = float(waveform_config.get("offset", 0))  # units? mA? optional

    def calculate(self, t):  # TODO better name
        if self.is_active(t):
            f = self.freq / 1000. # Hz to mHz
            a = self.amp
            return a * np.sin(2 * np.pi * f * t + self.phase_offset) + self.amp_offset
        else:
            return 0


class WaveformCustom(BaseWaveform):
    """Custom waveform defined by csv file"""
    def __init__(self, waveform_file):
        self.definition = pd.read_csv(waveform_file, sep='\t')

    def calculate(self, t):
        return np.interp(t, self.definition["time"], self.definition["amplitude"])


class ComplexWaveform(BaseWaveform):
    """Superposition of simple waveforms"""
    def __init__(self, el_collection):
        self.electrodes = el_collection

    def calculate(self, t):
        val = 0
        for el in self.electrodes:
            val += el.calculate(t)

        return val


# mapping from 'shape' code to subclass, always lowercase
shape_classes = {
    'dc': WaveformTypeDC,
    'sin': WaveformTypeSin,
}


def stimx_waveform_factory(waveform):
    """
    Factory to create correct waveform class based on conf.
    Supports json config in conf as well as string pointer to a file.
    :rtype: BaseWaveformType
    """
    if isinstance(waveform, string_types):
        # if waveform_conf is str or unicode assume to be name of file in stim_dir
        # waveform_conf = str(waveform_conf)   # make consistent
        file_ext = os.path.splitext(waveform)
        if file_ext == 'csv':
            return WaveformCustom(waveform)

        elif file_ext == 'json':
            with open(waveform, 'r') as f:
                waveform = json.load(f)
        else:
            io.log_warning('Unknwon filetype for waveform')

    shape_key = waveform["shape"].lower()

    if shape_key not in shape_classes:
        io.log_warning("Waveform shape not known")  # throw error?

    Constructor = shape_classes[shape_key]
    return Constructor(waveform)


def iclamp_waveform_factory(conf):
    """
    Factory to create correct waveform class based on conf.
    Supports json config in conf as well as string pointer to a file.
    :rtype: BaseWaveformType
    """
    iclamp_waveform_conf = conf["iclamp"]

    shape_key = iclamp_waveform_conf["shape"].lower()

    if shape_key not in shape_classes:
        io.log_warning('iclamp waveform shape not known')  # throw error?

    Constructor = shape_classes[shape_key]
    return Constructor(iclamp_waveform_conf)
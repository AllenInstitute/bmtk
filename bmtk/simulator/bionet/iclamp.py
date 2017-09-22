from neuron import h


class IClamp(object):
    def __init__(self, conf):
        self._conf = conf
        self._iclamp_amp = self._conf['iclamp']['amp']
        self._iclamp_del = self._conf['iclamp']['del']
        self._iclamp_dur = self._conf['iclamp']['dur']

    def attach_current(self, cell):
        self.cell = cell
        self.stim = h.IClamp(self.cell.hobj.soma[0](0.5))
        self.stim.delay = self._iclamp_del
        self.stim.dur = self._iclamp_dur
        self.stim.amp = self._iclamp_amp
        return self.stim
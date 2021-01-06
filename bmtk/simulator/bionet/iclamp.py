# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from neuron import h


class IClamp(object):
    def __init__(self, amplitude, delay, duration):
        self._iclamp_amp = amplitude
        self._iclamp_del = delay
        self._iclamp_dur = duration
        self._stim = None

    def attach_current(self, cell):
        self._stim = h.IClamp(cell.hobj.soma[0](0.5))
        self._stim.delay = self._iclamp_del
        self._stim.dur = self._iclamp_dur
        self._stim.amp = self._iclamp_amp
        return self._stim


class FileIClamp(object):
    def __init__(self, amplitudes, dt):
        self._iclamp_amps = amplitudes
        self._iclamp_dt = dt
        self._stim = None

    def attach_current(self, cell):
        self._stim = h.IClamp(cell.hobj.soma[0](0.5))

        # Listed as necessary values in the docs to use play() with an IClamp.
        self._stim.delay = 0
        self._stim.dur = 1e9

        self._vect_stim = h.Vector(self._iclamp_amps)
        self._vect_stim.play(self._stim._ref_amp, self._iclamp_dt)  #Plays the amps to the IClamp amp variable with a given dt.

        return self._stim


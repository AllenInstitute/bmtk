# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 20XX. Allen Institute. All rights reserved.
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
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import math
import pandas as pd


class RecXElectrode(object):
    """Extracellular electrode

    """
    def __init__(self,conf):
        """Create an array"""
        self.conf = conf
        electrode_file = self.conf["recXelectrode"]["positions"]       

        el_df = pd.read_csv(electrode_file,sep=' ')
        self.pos = el_df.as_matrix(columns=['x_pos','y_pos','z_pos']).T      # convert coordinates to ndarray, The first index is xyz and the second is the channel number
        self.nsites = self.pos.shape[1]
        self.conf['run']['nsites'] = self.nsites  # add to the config
        self.transfer_resistances = {}   # V_e = transfer_resistance*Im

    def drift(self):
        # will include function to model electrode drift
        pass
    
    def get_transfer_resistance(self, gid):
        return self.transfer_resistances[gid]
    
    def calc_transfer_resistance(self, gid, seg_coords):
        """Precompute mapping from segment to electrode locations"""
        sigma = 0.3  # mS/mm

        r05 = (seg_coords['p0'] + seg_coords['p1'])/2
        dl = seg_coords['p1'] - seg_coords['p0']
        
        nseg = r05.shape[1]
        
        tr = np.zeros((self.nsites,nseg))

        for j in xrange(self.nsites):   # calculate mapping for each site on the electrode
            rel = np.expand_dims(self.pos[:, j], axis=1)   # coordinates of a j-th site on the electrode
            rel_05 = rel - r05  # distance between electrode and segment centers
            r2 = np.einsum('ij,ij->j', rel_05, rel_05)    # compute dot product column-wise, the resulting array has as many columns as original
            
            rlldl = np.einsum('ij,ij->j', rel_05, dl)    # compute dot product column-wise, the resulting array has as many columns as original
            dlmag = np.linalg.norm(dl, axis=0)  # length of each segment
            rll = abs(rlldl/dlmag)   # component of r parallel to the segment axis it must be always positive
            rT2 = r2 - rll**2  # square of perpendicular component
            up = rll + dlmag/2
            low = rll - dlmag/2
            num = up + np.sqrt(up**2 + rT2)
            den = low + np.sqrt(low**2 + rT2)
            tr[j, :] = np.log(num/den)/dlmag  # units of (um) use with im_ (total seg current)

        tr *= 1/(4*math.pi*sigma)
        self.transfer_resistances[gid] = tr

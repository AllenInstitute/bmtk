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
import os
import numpy as np
from bmtk.simulator.utils import nwb
import pickle
import re

pickle_regex = re.compile('.*\.pkl')

def convert_filters(src_dir, tgt_dir):

    for file_name in os.listdir(src_dir):
        if not pickle_regex.match(file_name) is None:
            
            print('Converting: %s' % file_name)
            
            full_path_to_src_file = os.path.join(src_dir, file_name)
            full_path_to_tgt_file = os.path.join(tgt_dir, file_name).replace('.pkl', '.nwb')
            
            try:
                f = nwb.NWB(file_name=full_path_to_tgt_file,
                            identifier='iSee example filter dataset',
                            description='Convering an example inhomogenous Poisson rate collection from a filter to drive simulations')
                
                # Load data from file:
                data = pickle.load(open(full_path_to_src_file, 'r'))
                timestamps = data['t']
                 
                # Load first cell into file:
                ts0 = f.create_timeseries('TimeSeries', "Cell_0", "acquisition")
                ts0.set_data(data['cells'][0], unit='Hz', resolution=float('nan'), conversion=1.)
                ts0.set_time_by_rate(0.,1000.)
                ts0.set_value('num_samples', len(timestamps))
                ts0.finalize()
                 
                # Load remaining cells into file, linking timestamps:
                for ii in np.arange(1,len(data['cells'])):
                    ts = f.create_timeseries('TimeSeries', "Cell_%s" % ii, "acquisition")
                    ts.set_data(data['cells'][ii], unit='Hz', resolution=float('nan'), conversion=1.)
                    ts.set_time_by_rate(0.,1000.)
                    ts.set_value('num_samples', len(timestamps))
                    ts.finalize()
                 
                # Close out:
                f.close()
                
            except:
                print ('    Conversion failed: %s' % file_name)



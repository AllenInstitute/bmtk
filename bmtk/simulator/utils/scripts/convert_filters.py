import os
import numpy as np
from bmtk.simulator.utils import nwb
import pickle
import re

pickle_regex = re.compile('.*\.pkl')

def convert_filters(src_dir, tgt_dir):

    for file_name in os.listdir(src_dir):
        if not pickle_regex.match(file_name) is None:
            
            print 'Converting: %s' % file_name
            
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
                print '    Conversion failed: %s' % file_name



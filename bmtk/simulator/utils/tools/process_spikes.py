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
import h5py
import numpy as np
import pandas as pd
import os


def read_spk_txt(f_name):

    '''
    
    Parameters
    ----------
    f_name: string
        Full path to a file containing cell IDs and spike times.

    Returns
    -------
    A dataframe containing two columns: spike times and cell IDs.

    Usage:
    x = read_spk_txt('output/spk.dat')

    '''

    df = pd.read_csv(f_name, header=None, sep=' ')
    df.columns = ['t', 'gid']

    return df


def read_spk_h5(f_name):

    '''
    
    Parameters
    ----------
    f_name: string
        Full path to a file containing cell IDs and spike times.

    Returns
    -------
    A dataframe containing two columns: spike times and cell IDs.

    Usage:
    x = read_spk_h5('output/spk.h5')

    '''

    f = h5py.File(f_name, 'r' , libver='latest')
    spikes = {}

    t = np.array([])
    gids = np.array([])
    for i, gid in enumerate(f.keys()):  # save spikes of all gids
        if (i % 1000 == 0):
            print(i)
        spike_times = f[gid][...]
        t = np.append(t, spike_times)
        gids = np.append(gids, np.ones(spike_times.size)*int(gid))

    f.close()

    df = pd.DataFrame(columns=['t', 'gid'])
    df['t'] = t
    df['gid'] = gids

    return df


def spikes_to_mean_f_rate(cells_f, spk_f, t_window, **kwargs):

    '''
    
    Parameters
    ----------
    cells_f: string
        Full path to a file containing information about all cells (in particular, all cell IDs,
        and not just those that fired spikes in a simulation).
    spk_f: string
        Full path to a file containing cell IDs and spike times.
    t_window: a tuple of two floats
        Start and stop time for the window within which the firing rate is computed.
    **kwargs
    spk_f_type: string with accepted values 'txt' or 'h5'
        Type of the file from which spike times should be extracted.
    

    Assumptions
    -----------
    It is assumed here that TIME IS in ms and the RATES ARE RETURNED in Hz.
    

    Returns
    -------
    A dataframe containing a column of cell IDs and a column of corresponding
    average firing rates.

    Usage:
    x = spikes_to_mean_f_rate('../network_model/cells.csv', 'output/spk.dat', (500.0, 3000.0))

    '''

    # Make sure the time window's start and stop times are reasonable.
    t_start = t_window[0]
    t_stop = t_window[1]
    delta_t = t_stop - t_start
    if (delta_t <= 0.0):
        print('spikes_to_mean_f_rate: stop time %f is <= start time %f; exiting.' % (t_stop, t_start))
        quit()

    # Read information about all cells.
    cells_df = pd.read_csv(cells_f, sep=' ')
    gids = cells_df['id'].values

    # By default, the spk file type is "None", in which case it should be chosen
    # based on the extension of the supplied spk file name.
    spk_f_type = kwargs.get('spk_f_type', None)
    if (spk_f_type == None):
        spk_f_ext = spk_f.split('.')[-1]
        if (spk_f_ext in ['txt', 'dat']):
            spk_f_type = 'txt' # Assume this is an ASCII file.
        elif (spk_f_ext in ['h5']):
            spk_f_type = 'h5' # Assume this is an HDF5 file.
        else:
            print('spikes_to_mean_f_rate: unrecognized file extension.  Use the flag spk_f_type=\'txt\' or \'h5\' to override this message.  Exiting.')
            quit()

    # In case the spk_f_type was provided directly, check that the value is among those the code recognizes.
    if (spk_f_type not in ['txt', 'h5']):
        print('spikes_to_mean_f_rate: unrecognized value of spk_f_type.  The recognized values are \'txt\' or \'h5\'.  Exiting.')
        quit()

    # Read spikes.
    # If the spike file has zero size, create a dataframe with all rates equal to zero.
    # Otherwise, use spike times from the file to fill the dataframe.
    if (os.stat(spk_f).st_size == 0):
        f_rate_df = pd.DataFrame(columns=['gid', 'f_rate'])
        f_rate_df['gid'] = gids
        f_rate_df['f_rate'] = np.zeros(gids.size)
    else:
        # Use the appropriate function to read the spikes.
        if (spk_f_type == 'txt'):
            df = read_spk_txt(spk_f)
        elif(spk_f_type == 'h5'):
            df = read_spk_h5(spk_f)

        # Keep only those entries that have spike times within the time window.
        df = df[(df['t'] >= t_start) & (df['t'] <= t_stop)]

        # Compute rates.
        f_rate_df = df.groupby('gid').count() * 1000.0 / delta_t # Time is in ms and rate is in Hz.
        f_rate_df.columns = ['f_rate']
        # The 'gid' label is now used as index (after the groupby operation).
        # Convert it to a column; then change the index name to none, as in default.
        f_rate_df['gid'] = f_rate_df.index
        f_rate_df.index.names = ['']

        # Find cell IDs from the spk file that are not in the cell file.
        # Remove them from the dataframe with rates.
        gids_not_in_cells_f = f_rate_df['gid'].values[~np.in1d(f_rate_df['gid'].values, gids)]
        f_rate_df = f_rate_df[~f_rate_df['gid'].isin(gids_not_in_cells_f)]

        # Find cell IDs from the cell file that do not have counterparts in the spk file
        # (for example, because those cells did not fire).
        # Add these cell IDs to the dataframe; fill rates with zeros.
        gids_not_in_spk = gids[~np.in1d(gids, f_rate_df['gid'].values)]
        f_rate_df = f_rate_df.append(pd.DataFrame(np.array([gids_not_in_spk, np.zeros(gids_not_in_spk.size)]).T, columns=['gid', 'f_rate']))

        # Sort the rows according to the cell IDs.
        f_rate_df = f_rate_df.sort('gid', ascending=True)

    return f_rate_df


# Tests.

#x = spikes_to_mean_f_rate('/data/mat/yazan/corticalCol/ice/sims/column/build/net_structure/cells.csv', '/data/mat/yazan/corticalCol/ice/sims/column/full_preliminary_runs/output008/spikes.txt', (500.0, 2500.0))
#print x

#x = spikes_to_mean_f_rate('/data/mat/yazan/corticalCol/ice/sims/column/build/net_structure/cells.csv', '/data/mat/yazan/corticalCol/ice/sims/column/full_preliminary_runs/output008/spikes.h5', (500.0, 2500.0))
#print x

#x = spikes_to_mean_f_rate('/data/mat/yazan/corticalCol/ice/sims/column/build/net_structure/cells.csv', '/data/mat/yazan/corticalCol/ice/sims/column/full_preliminary_runs/output008/spikes.txt', (500.0, 2500.0), spk_f_type='txt')
#print x


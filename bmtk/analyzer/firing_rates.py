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
import pandas as pd
import numpy as np

def convert_rates(rates_file):
    rates_df = pd.read_csv(rates_file, sep=' ', names=['gid', 'time', 'rate'])
    rates_sorted_df = rates_df.sort_values(['gid', 'time'])
    rates_dict = {}
    for gid, rates in rates_sorted_df.groupby('gid'):
        start = rates['time'].iloc[0]
        #start = rates['rate'][0]
        end = rates['time'].iloc[-1]
        dt = float(end - start)/len(rates)
        rates_dict[gid] = {'start': start, 'end': end, 'dt': dt, 'rates': np.array(rates['rate'])}

    return rates_dict


def firing_rates_equal(rates_file1, rates_file2, err=0.0001):
    trial_1 = convert_rates(rates_file1)
    trial_2 = convert_rates(rates_file2)
    if set(trial_1.keys()) != set(trial_2.keys()):
        return False

    for gid, rates_data1 in trial_1.items():
        rates_data2 = trial_2[gid]
        if rates_data1['dt'] != rates_data2['dt'] or rates_data1['start'] != rates_data2['start'] or rates_data1['end'] != rates_data2['end']:
            return False

        for r1, r2 in zip(rates_data1['rates'], rates_data2['rates']):
            if abs(r1 - r2) > err:
                return False

    return True
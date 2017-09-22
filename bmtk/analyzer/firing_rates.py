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
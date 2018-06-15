import pandas as pd
import csv

class RatesInput(object):
    def __init__(self, params):
        self._rates_df = pd.read_csv(params['rates'], sep=' ')
        self._node_population = params['node_set']
        self._rates_dict = {int(row['gid']): row['firing_rate'] for _, row in self._rates_df.iterrows()}

    @property
    def populations(self):
        return [self._node_population]

    def get_rate(self, gid):
        return self._rates_dict[gid]


class RatesWriter(object):
    def __init__(self, file_name):
        self._file_name = file_name
        self._fhandle = open(file_name, 'a')
        self._csv_writer = csv.writer(self._fhandle, delimiter=' ')

    def add_rates(self, gid, times, rates):
        for t, r in zip(times, rates):
            self._csv_writer.writerow([gid, t, r])
        self._fhandle.flush()

    def to_csv(self, file_name):
        pass

    def to_h5(self, file_name):
        raise NotImplementedError



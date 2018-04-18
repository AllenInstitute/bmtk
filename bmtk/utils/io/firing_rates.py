import pandas as pd

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

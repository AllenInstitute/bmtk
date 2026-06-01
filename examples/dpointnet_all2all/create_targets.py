import h5py
import numpy as np
import pandas as pd


def create_individual_targets():
    with h5py.File('network/glifs_nodes.h5', 'r') as h5:
        node_ids = h5['/nodes/glifs/node_id'][()]

    firing_rates = np.linspace(1.0, 50.0, num=len(node_ids), endpoint=True)
    pd.DataFrame({
        'node_id': node_ids,
        'population': 'glifs',
        'firing_rate': firing_rates
    }).to_csv('target_firing_rates.individual.csv', index=False)


if __name__ == '__main__':
    create_individual_targets()
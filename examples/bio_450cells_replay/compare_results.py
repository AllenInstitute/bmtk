import os
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np

# from bmtk.analyzer.spike_trains import plot_raster

def get_nodes_table():
    node_types_df = pd.read_csv('network/internal_node_types.csv', sep=' ')
    nodes_h5 = h5py.File('network/internal_nodes.h5', 'r')
    node_ids_df = pd.DataFrame({
        'node_ids': nodes_h5['/nodes/internal/node_id'][()],
        'node_type_id': nodes_h5['/nodes/internal/node_type_id'][()]
    })
    return node_ids_df.merge(node_types_df, how='left', on='node_type_id')




def plot_rasters(output_dirs, titles=None):
    nodes_df = get_nodes_table()
    nodes_df = nodes_df[['node_ids', 'model_name']]
    y_range = [nodes_df['node_ids'].min()-1, nodes_df['node_ids'].max()+1]

    output_dirs = list(output_dirs) if not isinstance(output_dirs, (tuple, list)) else output_dirs
    n_rows = len(output_dirs)
    n_cols = 1
    if titles is not None:
        assert(len(titles) == n_rows)

    color_map = {} # makes sure all
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 1 + n_rows*2))
    axes = np.array([axes]) if n_rows == 1 else axes
    for idx, output_dir in enumerate(output_dirs):
        spikes_h5 = h5py.File(os.path.join(output_dir, 'spikes.h5'), 'r')
        spikes_df = pd.DataFrame({
            'node_ids': spikes_h5['/spikes/internal/node_ids'][()],
            'timestamps': spikes_h5['/spikes/internal/timestamps'][()]
        })
        spikes_df = spikes_df.merge(nodes_df, how='left', on='node_ids')
        for model_name, model_df in spikes_df.groupby('model_name'):
            if model_name not in color_map:
                sc = axes[idx].scatter(model_df['timestamps'], model_df['node_ids'], s=3, label=model_name)
                color_map[model_name] = sc.get_facecolors()[0]
            else:
                axes[idx].scatter(model_df['timestamps'], model_df['node_ids'], color=color_map[model_name], s=3, label=model_name)

        axes[idx].set_ylim(y_range)
        axes[idx].set_xlim([0.0, 3000.0])
        if idx != (n_rows-1):
            axes[idx].set_xticks([])

        if titles is not None:
            axes[idx].set_ylabel(titles[idx])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_rasters(output_dirs=['output_prev'])
    # plot_rasters(
    #     output_dirs=['output_feedforward', 'output_full', 'output_allcells'],
    #     titles=['feedforward inputs only', 'fully-recurrent', 'recurrent activity only']
    # )
    plot_rasters(
        output_dirs=['output_feedforward', 'output_full', 'output_allcells', 'output_wextern'],
        titles=['feedforward inputs only', 'fully-recurrent', 'disconnected inputs only', 'feedforward + disconnected']
    )
    # plot_rasters(output_dirs=['output_feedforward', 'output_full', 'output_allcells', 'output_wextern'])


#
# plot_raster(config_file='config.recurrent.json', group_by='model_name', show=False)
# plt.show()
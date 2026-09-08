import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# remove_mtypes = ['sON_TF4', 'sON_TF8', 'sOFF_TF4', 'sOFF_TF8', 'sOFF_TF15', 'tON_TF8', 'tON_TF15', 'tOFF_TF15', 'sON_TF1', 'sOFF_TF1', 'tOFF_TF8', 'sONtOFF_001']
remove_mtypes = [
    'tON_TF8', 'tON_TF15',
    'sON_TF2', 'sON_TF4', 'sON_TF1',
    'sOFF_TF1', 'sOFF_TF4', 'sOFF_TF15', 'sOFF_TF8',
    'tOFF_TF15', 'tOFF_TF8'
]

# model_templates = ['sON_TF2', 'sOFF_TF2', 'tON_TF4', 'tOFF_TF4', 'sONsOFF_001', 'sONtOFF_001']
model_templates = ['sON_TF8', 'sOFF_TF2', 'tON_TF4', 'tOFF_TF4', 'sONsOFF_001', 'sONtOFF_001']

nodes_h5 = h5py.File('network/lgn_nodes.h5', 'r')
nodes_df = pd.DataFrame({
    'node_id': nodes_h5['/nodes/lgn/node_id'][()],
    'node_type_id': nodes_h5['/nodes/lgn/node_type_id'][()]
})

node_types_df = pd.read_csv('network/lgn_node_types.csv', sep=' ')
node_types_df = node_types_df[['node_type_id', 'model_template', 'subtypex']]
node_types_df['model_template'] = node_types_df.apply(
    lambda r: r['model_template'].replace('lgnmodel:', ''),
    axis=1
)
nodes_df = pd.merge(nodes_df, node_types_df, how='left', on='node_type_id')

if remove_mtypes:
    nodes_df = nodes_df[~nodes_df['model_template'].isin(remove_mtypes)]
# if display_mtypes:
#     nodes_df = nodes_df[nodes_df['model_template'].isin(display_mtypes)]


if not model_templates:
    model_templates = nodes_df['model_template'].values
n_rows = len(model_templates)

output_rates_files = ['output_on_flash/rates.h5', 'output_off_flash/rates.h5']
n_cols = len(output_rates_files)


fig, axes = plt.subplots(n_rows+1, n_cols, figsize=(11, 8))

ts = np.linspace(0.0, 3.0, num=200)
on_flash = np.array([1.0 if 1.0 < t < 2.0 else 0.0 for t in ts])
axes[0, 0].plot(on_flash, c='k')
axes[0, 0].set_ylim(-1.1, 1.1)
axes[0, 0].axis('off')
axes[0, 0].set_title('ON Flash')
axes[0, 0].text(0.0, -0.35, 'brightness', size='x-small')

off_flash = np.array([-1.0 if 1.0 < t < 2.0 else 0.0 for t in ts])
axes[0, 1].plot(off_flash, c='k')
axes[0, 1].set_ylim(-1.1, 1.1)
axes[0, 1].axis('off')
axes[0, 1].set_title('OFF Flash')

# plt.show()
# exit()
# plt.plot()


max_fr_vals = [0 for _ in range(n_rows)]
for c, rates_file in enumerate(output_rates_files):
    
    rates_h5 = h5py.File(rates_file, 'r')
    node_ids_lu = rates_h5['firing_rates/lgn/node_id']
    timestamps = rates_h5['firing_rates/lgn/times'][()]
    firing_rates_ds = rates_h5['firing_rates/lgn/firing_rates_Hz']

    for r, mtemplate in enumerate(model_templates):
        model_grp = nodes_df[nodes_df['model_template'] == mtemplate]
        # print(model_grp)
        # print(mtemplate)
        # print(nodes_df)
        subtype = model_grp['subtypex'].iloc[0]
        node_idxs = node_ids_lu[model_grp['node_id'].values]
        # print(mtemplate, node_idxs)
        firing_rates = firing_rates_ds[:, node_idxs]
        firing_rate_avg = np.mean(firing_rates, axis=1)
        axes[r+1, c].plot(timestamps, firing_rate_avg)

        max_fr = np.max(firing_rate_avg)
        max_fr_vals[r] = max(max_fr_vals[r], max_fr)
   
        if c == 0:
            # axes[r, c].set_ylabel(f'{mtemplate}', rotation=0, size='large', labelpad=40)
            # axes[r, c].annotate(f'{mtemplate}', xy=(0.0, 0.5))
            axes[r+1, c].set_ylabel('rate (Hz)', size='small')
            pad = 5
            axes[r+1, c].annotate(
                subtype, 
                xy=(0, 0.5), xytext=(-axes[r+1, c].yaxis.labelpad - pad, 0),
                xycoords=axes[r+1, c].yaxis.label, textcoords='offset points',
                size='x-large', ha='right', va='center'
            )
        else:
            axes[r+1, c].set_yticks([])

        if r+1 != n_rows:
            axes[r+1, c].set_xticks([])
        else:
            axes[r+1, c].set_xlabel('time (s)')


for r in range(n_rows):
    axes[r+1, 0].set_ylim(-1.0, max_fr_vals[r]*1.3)
    axes[r+1, 1].set_ylim(-1.0, max_fr_vals[r]*1.3)
    

plt.tight_layout()
plt.show()
# exit()




# rates_h5 = h5py.File('output_on_flash/rates.h5', 'r')
# node_ids_lu = rates_h5['firing_rates/lgn/node_id']
# timestamps = rates_h5['firing_rates/lgn/times'][()]
# firing_rates_ds = rates_h5['firing_rates/lgn/firing_rates_Hz']






# for model_template, model_grp in nodes_df.groupby('model_template'):
#     # print(model_template, model_grp['node_id'].values)
#     node_idxs = node_ids_lu[model_grp['node_id'].values]
#     firing_rates = firing_rates_ds[:, node_idxs]
#     firing_rate_avg = np.mean(firing_rates, axis=1)
#     plt.plot(timestamps, firing_rate_avg, label=model_template)


# plt.legend()
# plt.show()
# exit()

# print(nodes_df)
# exit()



# for node_idx, node_id in enumerate(node_ids):
#     node_frs = firing_rates_ds[:, node_idx]
#     # print(node_frs)
#     plt.plot(timestamps, node_frs)
#     # break

# # print(times)
# # rates_h5['firing_rates/lgn/firing_rates_Hz']
# plt.show()



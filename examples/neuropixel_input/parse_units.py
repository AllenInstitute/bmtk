from pathlib import Path
import pynwb
import pandas as pd


pd.set_option('display.max_columns', None)

nwb_path = './ecephys_cache_dir/session_715093703/session_715093703.nwb'
file_dir = Path(__file__).parent
namespace_path = (file_dir / "ndx-aibs-ecephys.namespace.yaml").resolve()
pynwb.load_namespaces(str(namespace_path))
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwbfile = io.read()

units = nwbfile.units.to_dataframe()
units = units.rename(columns={'peak_channel_id': 'channel_id'})
# units = units.reset_index()
# units = units.rename(columns={'waveform_mean': 'mean_waveforms'})

print(units.shape)


channels = nwbfile.electrodes.to_dataframe()
channels = channels.reset_index().rename(columns={'id': 'channel_id'})

units = units.merge(channels, how='left', on='channel_id')
print(units.shape)
print(units['location'].value_counts())
# print(channels.reset_index().rename(columns={'id': 'channel_id'}).head())


# probe_names = [probe_name for probe_name in nwbfile.electrode_groups]
# channels = nwbfile.electrodes.to_dataframe()
# for probe_name in probe_names:
#     probe = nwbfile.electrode_groups[probe_name]
#     probe_id = probe.probe_id
#     # print(nwbfile.electrodes.to_dataframe())
#     # print(channels['probe_id'])
#     # print(probe_name, probe_id)
#     # print(channels[channels['probe_id'] == probe_id])
#     print(channels)


#     print(units['peak_channel_id'].unique())
#     print(channels[['probe_id', 'location']])
#     # print(units['peak_channel_id'].map({}))
#     exit()





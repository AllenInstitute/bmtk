from pathlib import Path
import pandas as pd
import pynwb
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


pd.set_option('display.max_columns', None)

nwb_path = './ecephys_cache_dir/session_715093703/session_715093703.nwb'


file_dir = Path(__file__).parent
namespace_path = (file_dir / "ndx-aibs-ecephys.namespace.yaml").resolve()
pynwb.load_namespaces(str(namespace_path))
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()

### Reading unit spike times
"""
# print(nwb.units.spike_times.data[()])
# print(nwb.units.spike_times[2])
print(nwb.units['spike_times'][0])
print(nwb.units['spike_times'][0:3])
print(nwb.units['id'][1])
"""

### Reading stimulus table


session = EcephysSession.from_nwb_path(nwb_path)
# print(session.units.head())
# print(session.units['structure_acronym'].value_counts())
# print(session.get_stimulus_table())

# print(session.units)
# print(session.spike_times[950910352])
stims = session.stimulus_presentations
dg_stims = stims[
    (stims['stimulus_name'] == 'drifting_gratings') 
    & (stims['temporal_frequency'] == 4.0)
    & (stims['stimulus_block'] == 2.0)
    & (stims['orientation'].replace('null', np.nan).astype(float) > 310.0)
]
# print(stims['orientation'].replace('null', np.nan).astype(float))
print(dg_stims)


# print(nwb.intervals.items())
# for stim_name, interval in nwb.intervals.items():
#     print('--------', stim_name, '-----------')
#     print(interval.to_dataframe())
#     # print(stim_name, interval)



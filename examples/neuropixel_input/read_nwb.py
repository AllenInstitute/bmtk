from pathlib import Path
import pandas as pd
import pynwb
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


pd.set_option('display.max_columns', None)


# file_dir = Path(__file__).parent
# namespace_path = (file_dir / "ndx-aibs-ecephys.namespace.yaml").resolve()
# pynwb.load_namespaces(str(namespace_path))
# io = pynwb.NWBHDF5IO('/local1/workspace/neuropixels/ecephys_cache_dir/session_759883607/session_759883607.nwb', 'r')
# nwb = io.read()

### Reading unit spike times
"""
# print(nwb.units.spike_times.data[()])
# print(nwb.units.spike_times[2])
print(nwb.units['spike_times'][0])
print(nwb.units['spike_times'][0:3])
print(nwb.units['id'][1])
"""

### Reading stimulus table

nwb_path = './ecephys_cache_dir/session_715093703/session_715093703.nwb'
session = EcephysSession.from_nwb_path(nwb_path)
print(session.units.head())
print(session.units['structure_acronym'].value_counts())
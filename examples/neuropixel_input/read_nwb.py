from pathlib import Path
import pynwb

file_dir = Path(__file__).parent
namespace_path = (file_dir / "ndx-aibs-ecephys.namespace.yaml").resolve()

pynwb.load_namespaces(str(namespace_path))

io = pynwb.NWBHDF5IO('/local1/workspace/neuropixels/ecephys_cache_dir/session_759883607/session_759883607.nwb', 'r')
nwb = io.read()

### Reading unit spike times
"""
# print(nwb.units.spike_times.data[()])
# print(nwb.units.spike_times[2])
print(nwb.units['spike_times'][0])
print(nwb.units['spike_times'][0:3])
print(nwb.units['id'][1])
"""
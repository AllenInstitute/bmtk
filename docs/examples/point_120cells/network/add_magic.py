import h5py
import glob

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version

for h5file in glob.glob('*.h5'):
    with h5py.File(h5file, 'r+') as h5:
        add_hdf5_version(h5)
        add_hdf5_magic(h5)
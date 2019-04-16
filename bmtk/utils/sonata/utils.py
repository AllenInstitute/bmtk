# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import sys

import h5py
import pandas as pd
import numpy as np

MAGIC_ATTR = 'magic'
MAGIC_VAL = 0x0A7A
VERSION_ATTR = 'version'
VERSION_NA = 'NA'
VERSION_CURRENT = '0.1'

try:
    ver_split = VERSION_CURRENT.split('.')
    VERSION_MAJOR = ver_split[0]
    VERSION_MINOR = ver_split[1]
except (IndexError, AttributeError) as err:
    VERSION_MAJOR = 0
    VERSION_MINOR = 1


def listify(files):
    # TODO: change this to include any iterable datastructures (sets, panda sequences, etc)
    if not isinstance(files, (list, tuple)):
        return [files]
    else:
        return files


def load_h5(h5file, mode='r'):
    # TODO: Allow for h5py.Group also
    if isinstance(h5file, h5py.File):
        return h5file

    return h5py.File(h5file, mode)


def load_csv(csvfile):
    # TODO: make the separator more flexible
    if isinstance(csvfile, pd.DataFrame):
        return csvfile

    # TODO: check if it is csv object and convert to a pd dataframe
    return pd.read_csv(csvfile, sep=' ', na_values='NONE')


def get_attribute_h5(h5obj, attribut_name, default=None):
    val = h5obj.attrs.get(attribut_name, default)
    if using_py3 and isinstance(val, bytes):
        # There is an but with h5py returning unicode/str based attributes as bytes
        val = val.decode()

    return val


def check_magic(hdf5_file):
    """Check the magic attribute exists according to the sonata format"""
    h5_file_obj = load_h5(hdf5_file)
    if MAGIC_ATTR not in h5_file_obj.attrs:
        raise Exception('File {} missing top-level \"{}\" attribute.'.format(h5_file_obj.filename, MAGIC_ATTR))
    elif np.uint32(get_attribute_h5(hdf5_file, MAGIC_ATTR)) != MAGIC_VAL:
        raise Exception('File {} has unexpected magic value (expected {})'.format(h5_file_obj.filename, MAGIC_VAL))

    return True


def get_version(hdf5_file):
    h5_file_obj = load_h5(hdf5_file)
    if VERSION_ATTR not in h5_file_obj.attrs:
        return VERSION_NA

    else:
        version_val = get_attribute_h5(h5_file_obj, VERSION_ATTR)
        version_str = str(version_val[0])
        for ver_sub in version_val[1:]:
            version_str += '.{}'.format(ver_sub)
        return version_str


def add_hdf5_magic(hdf5_handle):
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)


def add_hdf5_version(hdf5_handle):
    hdf5_handle['/'].attrs['version'] = [np.uint32(VERSION_MAJOR), np.uint32(VERSION_MINOR)]


def get_node_ids(nodes_path, population):
    # Used by PoissonSpikesGenerator
    with h5py.File(nodes_path, 'r') as h5:
        node_ids = h5['/nodes'][population]['node_id'][()]
    return node_ids


if sys.version_info[0] == 3:
    using_py3 = True
    range_itr = range
else:
    using_py3 = False
    range_itr = xrange

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
import h5py
import pandas as pd


class ColumnProperty(object):
    """Representation of a column name and metadata from a hdf5 dataset, csv column, etc.

    """
    def __init__(self, name, dtype, dimension, nrows=0, attrs=None):
        self._name = name
        self._dtype = dtype
        self._dim = dimension
        self._nrows = nrows
        self._attrs = attrs or {}

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dimension(self):
        return self._dim

    @property
    def nrows(self):
        return self._nrows

    @property
    def attributes(self):
        return self._attrs

    @classmethod
    def from_h5(cls, hf_obj, name=None):
        if isinstance(hf_obj, h5py.Dataset):
            ds_name = name if name is not None else hf_obj.name.split('/')[-1]
            ds_dtype = hf_obj.dtype

            # If the dataset shape is in the form "(N, M)" then the dimension is M. If the shape is just "(N)" then the
            # dimension is just 1
            dim = 1 if len(hf_obj.shape) < 2 else hf_obj.shape[1]
            nrows = hf_obj.shape[0]
            return cls(ds_name, ds_dtype, dim, nrows, attrs=hf_obj.attrs)

        elif isinstance(hf_obj, h5py.Group):
            columns = []
            for name, ds in hf_obj.items():
                if isinstance(ds, h5py.Dataset):
                    columns.append(ColumnProperty.from_h5(ds, name))
            return columns

        else:
            raise Exception('Unable to convert hdf5 object {} to a property or list of properties.'.format(hf_obj))

    @classmethod
    def from_csv(cls, pd_obj, name=None):
        if isinstance(pd_obj, pd.Series):
            c_name = name if name is not None else pd_obj.name
            c_dtype = pd_obj.dtype
            return cls(c_name, c_dtype, 1)

        elif isinstance(pd_obj, pd.DataFrame):
            return [cls(name, pd_obj[name].dtype, 1) for name in pd_obj.columns]

        else:
            raise Exception('Unable to convert pandas object {} to a property or list of properties.'.format(pd_obj))

    def __hash__(self):
        return hash(self._name)

    def __repr__(self):
        return '{}'.format(self.name, self.dtype)

    def __eq__(self, other):
        if isinstance(other, ColumnProperty):
            return self._name == other._name
        else:
            return self._name == other

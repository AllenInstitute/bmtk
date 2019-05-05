import os
from collections import Counter
import numpy as np
import pandas as pd
import h5py


class SpikesFile(object):
    _file_adaptors = {}

    def __init__(self, filename, filetype=None, **params):
        self._ftype = self._get_file_type(filename, filetype)
        self._adaptor = SpikesFile._file_adaptors[self._ftype](filename, **params)

    def _get_file_type(self, filename, filetype):
        if filetype is not None:
            if filetype not in self._file_adaptors:
                raise Exception('Unknown spikes file type {}'.format(filetype))
            else:
                return filetype

        else:
            for ft, adaptor_cls in self._file_adaptors.items():
                if adaptor_cls.is_type(filename):
                    return ft

            raise Exception('Unable to determine file type for {}.'.format(filename))

    def _get_spikes_sort(self, spikes_list, t_window=None):
        if t_window is not None:
            spikes_list.sort()
            return [s for s in spikes_list if t_window[0] <= s <= t_window[1]]
        else:
            spikes_list.sort()
            return spikes_list

    @property
    def gids(self):
        """Return a list of all gids"""
        return self._adaptor.gids

    def to_dataframe(self):
        return self._adaptor.to_dataframe()

    def get_spikes(self, gid, time_window=None):
        return self._adaptor.get_spikes(gid, time_window=None)

    def __eq__(self, other):
        return self.is_equal(other)

    def is_equal(self, other, err=0.00001, time_window=None):
        # check that gids matches
        if set(self.gids) != set(other.gids):
            return False

        for gid in self.gids:
            spikes_self = self._get_spikes_sort(self.get_spikes(gid), time_window)
            spikes_other = self._get_spikes_sort(other.get_spikes(gid), time_window)

            if len(spikes_other) != len(spikes_self):
                return False

            for s0, s1 in zip(spikes_self, spikes_other):
                if abs(s0 - s1) > err:
                    return False
        return True

    @classmethod
    def register_adaptor(cls, adaptor_cls):
        cls._file_adaptors[adaptor_cls.ext_name()] = adaptor_cls
        return adaptor_cls


class SpikesFileAdaptor(object):
    def __init__(self, filename):
        self._filename = filename

    @property
    def gids(self):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError

    def get_spikes(self, gid, time_window=None):
        raise NotImplementedError

    @staticmethod
    def is_type(filename):
        raise NotImplementedError

    @staticmethod
    def ext_name():
        raise NotImplementedError


@SpikesFile.register_adaptor
class SpikesFileH5(SpikesFileAdaptor):
    def __init__(self, filename, **params):
        super(SpikesFileH5, self).__init__(filename)
        self._h5_handle = h5py.File(self._filename, 'r')
        self._sort_order = self._h5_handle['/spikes'].attrs.get('sorting', None)
        self._gid_ds = self._h5_handle['/spikes/gids']
        self._timestamps_ds = self._h5_handle['/spikes/timestamps']

        self._indexed = False
        self._gid_indicies = {}
        self._build_indicies()

    def _build_indicies(self):
        if self._sort_order == 'by_gid':
            indx_beg = 0
            c_gid = self._gid_ds[0]
            for indx, gid in enumerate(self._gid_ds):
                if gid != c_gid:
                    self._gid_indicies[c_gid] = slice(indx_beg, indx)
                    c_gid = gid
                    indx_beg = indx
            self._gid_indicies[c_gid] = slice(indx_beg, indx+1)
            self._indexed = True
        else:
            self._gid_indicies = {int(gid): [] for gid in np.unique(self._gid_ds)}
            for indx, gid in enumerate(self._gid_ds):
                self._gid_indicies[gid].append(indx)
            self._indexed = True

    @property
    def gids(self):
        return list(self._gid_indicies.keys())

    def to_dataframe(self):
        return pd.DataFrame({'timestamps': self._timestamps_ds, 'gids': self._gid_ds})

    def get_spikes(self, gid, time_window=None):
        return self._timestamps_ds[self._gid_indicies[gid]]

    @staticmethod
    def is_type(filename):
        _, fext = os.path.splitext(filename)
        fext = fext.lower()
        return fext == '.h5' or fext == '.hdf' or fext == '.hdf5'

    @staticmethod
    def ext_name():
        return 'h5'


@SpikesFile.register_adaptor
class SpikesFileCSV(SpikesFileAdaptor):
    def __init__(self, filename, **params):
        super(SpikesFileCSV, self).__init__(filename)
        self._spikes_df = pd.read_csv(self._filename, names=['timestamps', 'gids'], sep=' ')

    @property
    def gids(self):
        return list(self._spikes_df.gids.unique())

    def to_dataframe(self):
        return self._spikes_df

    def get_spikes(self, gid, time_window=None):
        return np.array(self._spikes_df[self._spikes_df.gids == gid].timestamps)

    @staticmethod
    def is_type(filename):
        _, fext = os.path.splitext(filename)
        fext = fext.lower()
        return fext == '.csv' or fext == '.txt'

    @staticmethod
    def ext_name():
        return 'csv'



import tensorflow as tf
from glob import glob
import numpy as np
import pickle as pkl
from pathlib import Path


class CachedInitState:
    def __init__(self, file_paths, file_type="mixed", **kwargs):
        self._rnn = kwargs.get("rnn")
        if file_paths is None:
            raise ValueError(
                f"CachedInitState: No file_paths specified, unable to load init_state."
            )

        file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self.all_cache_files = []
        for fp in file_paths:
            self.all_cache_files.extend(glob(fp))

        if len(self.all_cache_files) == 0:
            raise ValueError(
                f'{self.__class__} Could not find any cache files using file_paths="{file_paths}"'
            )

        self._load_fn = None
        if file_type is None or file_type in ["mixed", ""]:
            self._load_fn = CachedInitState._load_cache
        elif file_type in ["npz", "npy", "numpy", "memmap"]:
            self._load_fn = CachedInitState._load_npz
        elif file_type in ["pkl", "pickle"]:
            self._load_fn = CachedInitState._load_pkl
        elif file_type in ["h5", "hdf5"]:
            self._load_fn = CachedInitState._load_hdf5
        else:
            raise ValueError(
                f'{self.__class__}: Invalid file_type option "{file_type}". Available options [mixed, npz, pkl, h5].'
            )

    @staticmethod
    def _load_pkl(file_path, rnn):
        init_state = None
        with open(file_path, "wb") as f:
            init_state = pkl.load(file_path)
        return init_state

    @staticmethod
    def _load_npz(file_path, rnn):
        npz_data = np.load(file_path)

        state_vals = []
        _, state_names = rnn.cell.zero_state(
            batch_size=rnn.batch_size, dtype=rnn.dtype, with_names=True
        )
        for name in state_names:
            if name == "noise_step0" and name not in npz_data:
                state_vals.append(np.zeros((rnn.batch_size,), dtype=np.int32))
            else:
                state_vals.append(npz_data[name])

        return state_vals

    @staticmethod
    def _load_hdf5(file_path, rnn):
        raise NotImplementedError

    @staticmethod
    def _load_cache(file_path, rnn):
        ext = Path(file_path).suffix

        if ext in [".pickle", ".pkl", ".obj"]:
            return CachedInitState._load_pkl(file_path=file_path, rnn=rnn)
        elif ext in [".npz", ".npy", ".memmap"]:
            return CachedInitState._load_npz(file_path=file_path, rnn=rnn)
        elif ext in [".h5", ".hdf5"]:
            return CachedInitState._load_hdf5(file_path=file_path, rnn=rnn)

        try:
            return CachedInitState._load_pkl(file_path=file_path, rnn=rnn)
        except Exception as e:
            pass

        try:
            return CachedInitState._load_npz(file_path=file_path, rnn=rnn)
        except Exception as e:
            pass

        try:
            return CachedInitState._load_hdf5(file_path=file_path, rnn=rnn)
        except Exception as e:
            pass

        raise RuntimeError(f"Could not load {file_path}")

    def get_state(self, rnn=None, **kwargs):
        rnn = rnn or self._rnn
        if rnn is None:
            raise ValueError(
                "CachedInitState requires an RNN instance. Pass rnn to get_state() "
                "or construct CachedInitState with rnn=... ."
            )
        selected_file = np.random.choice(self.all_cache_files)
        if not Path(selected_file).exists():
            raise FileExistsError(f"Could not find init_state file {selected_file}")

        init_state = self._load_fn(selected_file, rnn)
        expected_state = rnn.cell.zero_state(batch_size=rnn.batch_size, dtype=rnn.dtype)
        if len(init_state) == len(expected_state) - 1:
            batch_size = tf.shape(init_state[0])[0]
            init_state = tuple(init_state) + (tf.zeros((batch_size,), dtype=tf.int32),)
        return init_state

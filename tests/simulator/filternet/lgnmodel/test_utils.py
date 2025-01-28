import pytest

from bmtk.simulator.filternet.lgnmodel import movie
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.util_fns import *


def test_get_tcross_from_temporal_kernel():
    mv = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))
    tf = TemporalFilterCosineBump(weights=[33.328, -2.10059], kpeaks=[59.0, 120.0], delays=[0.0, 0.0])
    tcross_ind = get_tcross_from_temporal_kernel(tf.get_kernel(threshold=-1.0).kernel)
    print(tcross_ind == 201)


if __name__ == '__main__':
    test_get_tcross_from_temporal_kernel()

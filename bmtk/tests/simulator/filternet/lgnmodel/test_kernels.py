import pytest
import numpy as np

from bmtk.simulator.filternet.lgnmodel.kernel import Kernel1D, Kernel2D, Kernel3D
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter
from bmtk.simulator.filternet.lgnmodel import movie


def test_spatialfilter_kernel():
    mv = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))

    gsf = GaussianSpatialFilter(translate=(-80, -20), sigma=(30, 10), rotation=15.0)
    kernel = gsf.get_kernel(row_range=mv.row_range, col_range=mv.col_range)
    assert(isinstance(kernel, Kernel2D))
    assert(kernel.full().shape == (120, 240))
    assert(np.isclose(np.sum(kernel.full()), 1.0))


def test_temporalfilter_kernel():
    mv = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))

    tf = TemporalFilterCosineBump(weights=[33.328, -2.10059], kpeaks=[59.0, 120.0],  delays=[0.0, 0.0])
    kernel = tf.get_kernel(t_range=mv.t_range, threshold=0.0, reverse=True)
    assert(isinstance(kernel, Kernel1D))
    assert(kernel.full().shape == (478,))
    assert(sum(kernel.full()) > 1.0)
    kernel.normalize()
    assert(np.isclose(sum(kernel.full()), 1.0))


def test_spatiotemporalfilter_kernel():
    mv = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))

    tf = TemporalFilterCosineBump(weights=[33.328, -2.10059], kpeaks=[59.0, 120.0], delays=[0.0, 0.0])
    sf = GaussianSpatialFilter(translate=(-80, -20), sigma=(30, 10), rotation=15.0)
    stf = SpatioTemporalFilter(sf, tf)
    kernel = stf.get_spatiotemporal_kernel(row_range=mv.row_range, col_range=mv.col_range, t_range=mv.t_range)
    assert(isinstance(kernel, Kernel3D))
    assert(kernel.full().shape == (987, 120, 240))
    kernel.normalize()
    assert(np.isclose(np.sum(kernel.full()), 1.0))


if __name__ == '__main__':
    # test_spatialfilter_kernel()
    # test_temporalfilter_kernel()
    test_spatiotemporalfilter_kernel()


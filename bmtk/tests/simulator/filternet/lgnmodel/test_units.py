import numpy as np
from sympy.abc import x as symbolic_x
from sympy.abc import y as symbolic_y

from bmtk.simulator.filternet.lgnmodel.cellmodel import OnUnit, OffUnit, LGNOnOffCell, TwoSubfieldLinearCell
from bmtk.simulator.filternet.lgnmodel.linearfilter import SpatioTemporalFilter
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump
from bmtk.simulator.filternet.lgnmodel.transferfunction import ScalarTransferFunction, MultiTransferFunction
from bmtk.simulator.filternet.lgnmodel.lgnmodel1 import LGNModel
from bmtk.simulator.filternet.lgnmodel import movie


def test_onunit():
    ffm = movie.FullFieldFlashMovie(range(120), range(240), 0.3, 0.7)
    mv = ffm.full(t_max=2.0)

    spatial_filter = GaussianSpatialFilter(translate=(120.0, 60.0), sigma=(0.615, 0.615), origin=(0.0, 0.0))
    temporal_filter = TemporalFilterCosineBump(weights=[3.441, -2.115], kpeaks=[8.269, 19.991], delays=[0.0, 0.0])
    linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=1.0)
    transfer_function = ScalarTransferFunction('Heaviside(s+1.05)*(s+1.05)')

    cell = OnUnit(linear_filter, transfer_function)
    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=10)
    assert(len(results) == 1)

    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.41666, 0.83333, 1.250, 1.6666], atol=1.0e-4))
    assert(np.allclose(rates, [1.05, 0.8635, 1.05, 1.05, 1.05], atol=1.0e-3))


def test_offunit():
    ffm = movie.FullFieldFlashMovie(range(120), range(240), 0.3, 0.7)
    mv = ffm.full(t_max=2.0)

    spatial_filter = GaussianSpatialFilter(translate=(120.0, 60.0), sigma=(0.615, 0.615), origin=(0.0, 0.0))
    temporal_filter = TemporalFilterCosineBump(weights=[3.441, -2.115], kpeaks=[8.269, 19.991], delays=[0.0, 0.0])
    linear_filter = SpatioTemporalFilter(spatial_filter, temporal_filter, amplitude=-1.0)
    transfer_function = ScalarTransferFunction('Heaviside(s+1.05)*(s+1.05)')

    cell = OffUnit(linear_filter, transfer_function)
    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=10)
    assert(len(results) == 1)

    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.41666, 0.83333, 1.250, 1.6666], atol=1.0e-4))
    assert(np.allclose(rates, [1.05, 1.2364, 1.05, 1.05, 1.05], atol=1.0e-3))


def test_lgnonoffcell():
    ffm = movie.FullFieldFlashMovie(range(120), range(240), 0.3, 0.7)
    mv = ffm.full(t_max=2.0)

    temporal_filter = TemporalFilterCosineBump(weights=[3.441, -2.115], kpeaks=[8.269, 19.991], delays=[0.0, 0.0])

    spatial_filter_on = GaussianSpatialFilter(sigma=(1.85, 1.85), origin=(0.0, 0.0), translate=(120.0, 60.0))
    on_linear_filter = SpatioTemporalFilter(spatial_filter_on, temporal_filter, amplitude=20)

    spatial_filter_off = GaussianSpatialFilter(sigma=(3.85, 3.85), origin=(0.0, 0.0), translate=(120.0, 60.0))
    off_linear_filter = SpatioTemporalFilter(spatial_filter_off, temporal_filter, amplitude=-20)

    cell = LGNOnOffCell(on_linear_filter, off_linear_filter)
    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=10)
    assert (len(results) == 1)

    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.41666, 0.83333, 1.250, 1.6666], atol=1.0e-4))
    assert(np.allclose(rates, [0.0, 3.7286, 0.0, 0.0, 0.0], atol=1.0e-3))


def test_twosubfieldlinearcell():
    ffm = movie.FullFieldFlashMovie(range(120), range(240), 0.3, 0.7)
    mv = ffm.full(t_max=2.0)

    spatial_filter = GaussianSpatialFilter(translate=(120.0, 60.0), sigma=(0.615, 0.615), origin=(0.0, 0.0))

    son_tfiler = TemporalFilterCosineBump([2.696143077048376, -1.8923936798453962],
                                          [37.993506826528716, 71.40822128514205], [42.0, 71.90456690180808])
    soff_tfilter = TemporalFilterCosineBump([3.7309552296292257, -1.4209858354384888],
                                            [21.556972532016253, 51.56392683711558], [61.0, 74.85742945288372])

    linear_filter_son = SpatioTemporalFilter(spatial_filter, son_tfiler, amplitude=1.0)
    linear_filter_soff = SpatioTemporalFilter(spatial_filter, soff_tfilter, amplitude=-1.51426850536)

    two_sub_transfer_fn = MultiTransferFunction((symbolic_x, symbolic_y),
                                                'Heaviside(x+2.0)*(x+2.0)+Heaviside(y+2.0)*(y+2.0)')

    cell = TwoSubfieldLinearCell(linear_filter_soff, linear_filter_son, subfield_separation=6.64946870229,
                                 onoff_axis_angle=249.09534316916634,
                                 dominant_subfield_location=(23.194207541958235, 49.44758663758982),
                                 transfer_function=two_sub_transfer_fn)

    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=10)
    assert (len(results) == 1)

    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.41666, 0.83333, 1.250, 1.6666], atol=1.0e-4))
    assert(np.allclose(rates, [4.0, 3.26931, 3.885, 4.0, 4.0], atol=1.0e-3))


if __name__ == '__main__':
    # test_onunit()
    # test_offunit()
    # test_lgnonoffcell()
    test_twosubfieldlinearcell()

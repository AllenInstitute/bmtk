import pytest
import numpy as np

from bmtk.simulator.filternet.lgnmodel.cellmodel import OffUnit, OnUnit, TwoSubfieldLinearCell
from bmtk.simulator.filternet.default_setters import default_cell_loader
from bmtk.simulator.filternet.lgnmodel.lgnmodel1 import LGNModel
from bmtk.simulator.filternet.lgnmodel import movie


class MockNode(object):
    node_params = {
        'x': 120.0,
        'y': 60.0,
        'spatial_size': 1.845
    }

    jitter = (1.0, 1.0)

    sf_sep = 6.0

    tuning_angle = 83.67

    weights = None
    kpeaks = None
    delays = None
    predefined_jitter = False

    weights_non_dom = None
    kpeaks_non_dom = None
    delays_non_dom = None

    non_dom_params = {
        'opt_wts': [3.59404059587911, -1.8145831206023941],
        'opt_delays': [0, 25],
        'opt_kpeaks': [30.758868538755, 58.35845979325414]
    }

    def get(self, key, default_val):
        if key in self.node_params:
            return self[key]
        else:
            return default_val

    def __getitem__(self, item):
        return self.node_params[item]

    def __contains__(self, item):
        return item in self.node_params

dynamics_params = {
    'opt_wts': [3.4416603571978417, -2.1155994819051305],
    'opt_kpeaks': [8.269733598229024, 19.99148791096526],
    'opt_delays': [0.0, 0.0]
}

np.set_printoptions(precision=4)


@pytest.mark.parametrize("cell_type,expected_val", [
    ('tON_TF8', [2.6, 2.7673, 2.7335, 2.5662, 2.4327]),
    ('sON_TF1', [5.2, 5.3673, 5.3335, 5.1662, 5.0327]),
    ('sON_TF2', [3.5333, 3.7006, 3.6668, 3.4995, 3.366]),
    ('sON_TF4', [8.25, 8.4173, 8.3835, 8.2162, 8.0827]),
    ('sON_TF8', [1.7667, 1.934, 1.9001, 1.7328, 1.5994]),
    ('sON_TF15', [10.88, 11.0473, 11.0135, 10.8462, 10.7127]),
])
def test_onunit(cell_type, expected_val):
    gm = movie.GratingMovie(row_size=120, col_size=240, frame_rate=24.0)
    mv = gm.create_movie(t_max=1.0)

    cell = default_cell_loader(MockNode(), ('lgnmodel', cell_type), dynamics_params)
    assert(isinstance(cell, OnUnit))

    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=5)  # Does the filtering + non-linearity on movie object m
    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.2083, 0.4167, 0.6250, 0.8333], atol=1.0e-3))
    assert(np.allclose(rates, expected_val, atol=1.0e-3))


@pytest.mark.parametrize("cell_type,expected_val", [
    ('tOFF_TF1', [5.5, 5.3327, 5.3665, 5.5338, 5.6673]),
    ('tOFF_TF2', [3.4095, 3.2422, 3.2761, 3.4434, 3.5768]),
    ('tOFF_TF4', [3.9446, 3.7773, 3.8111, 3.9784, 4.1118]),
    ('tOFF_TF8', [2.4378, 2.2705, 2.3043, 2.4716, 2.6051]),
    ('tOFF_TF15', [1.0507, 0.8834, 0.9172, 1.0845, 1.218]),
    ('sOFF_TF1', [3.95, 3.7827, 3.8165, 3.9838, 4.1173]),
    ('sOFF_TF2', [4.5318, 4.3645, 4.3983, 4.5656, 4.6991]),
    ('sOFF_TF4', [3.6791, 3.5118, 3.5456, 3.7129, 3.8464]),
    ('sOFF_TF8', [3.62, 3.4527, 3.4865, 3.6538, 3.7873]),
    ('sOFF_TF15', [5.075, 4.9077, 4.9415, 5.1088, 5.2423])
])
def test_offunit(cell_type, expected_val):
    gm = movie.GratingMovie(row_size=120, col_size=240, frame_rate=24.0)
    mv = gm.create_movie(t_max=1.0)

    cell = default_cell_loader(MockNode(), ('lgnmodel', cell_type), dynamics_params)
    assert(isinstance(cell, OffUnit))

    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=5)  # Does the filtering + non-linearity on movie object m
    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.2083, 0.4167, 0.6250, 0.8333], atol=1.0e-3))
    assert(np.allclose(rates, expected_val, atol=1.0e-3))


@pytest.mark.parametrize("cell_type,expected_val", [
    ('sONsOFF_001', [4.0, 5.6769, 5.2724, 3.5743, 2.3019])
])
def test_sONsOFF(cell_type, expected_val):
    gm = movie.GratingMovie(row_size=120, col_size=240, frame_rate=24.0)
    mv = gm.create_movie(t_max=1.0)

    cell = default_cell_loader(MockNode(), ('lgnmodel', cell_type), dynamics_params)
    assert(isinstance(cell, TwoSubfieldLinearCell))

    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=5)  # Does the filtering + non-linearity on movie object m
    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.2083, 0.4167, 0.6250, 0.8333], atol=1.0e-3))
    assert(np.allclose(rates, expected_val, atol=1.0e-3))


@pytest.mark.parametrize("cell_type,expected_val", [
    ('sONsOFF_001', [4.0, 5.6769, 5.2724, 3.5743, 2.3019])
])
def test_sONtOFF(cell_type, expected_val):
    gm = movie.GratingMovie(row_size=120, col_size=240, frame_rate=24.0)
    mv = gm.create_movie(t_max=1.0)

    cell = default_cell_loader(MockNode(), ('lgnmodel', cell_type), dynamics_params)
    assert(isinstance(cell, TwoSubfieldLinearCell))

    lgn = LGNModel([cell])
    results = lgn.evaluate(mv, downsample=5)  # Does the filtering + non-linearity on movie object m
    times = np.array(results[0][0], dtype=np.float64)
    rates = np.array(results[0][1], dtype=np.float64)

    assert(np.allclose(times, [0.0, 0.2083, 0.4167, 0.6250, 0.8333], atol=1.0e-3))
    assert(np.allclose(rates, expected_val, atol=1.0e-3))



if __name__ == '__main__':
    # test_offunit('tOFF_TF1', [5.5, 5.3327, 5.3665, 5.5338, 5.6673])
    # test_offunit('tOFF_TF2', [3.4095, 3.2422, 3.2761, 3.4434, 3.5768])
    # test_offunit('tOFF_TF4', [3.9446, 3.7773, 3.8111, 3.9784, 4.1118])
    # test_offunit('tOFF_TF8', [2.4378, 2.2705, 2.3043, 2.4716, 2.6051])
    # test_offunit('tOFF_TF15', [1.0507, 0.8834, 0.9172, 1.0845, 1.218])
    #
    # test_offunit('sOFF_TF1', [3.95, 3.7827, 3.8165, 3.9838, 4.1173])
    # test_offunit('sOFF_TF2', [4.5318, 4.3645, 4.3983, 4.5656, 4.6991])
    # test_offunit('sOFF_TF4', [3.6791, 3.5118, 3.5456, 3.7129, 3.8464])
    # test_offunit('sOFF_TF8', [3.62, 3.4527, 3.4865, 3.6538, 3.7873])
    # test_offunit('sOFF_TF15', [5.075, 4.9077, 4.9415, 5.1088, 5.2423])
    #
    # test_onunit('tON_TF8', [2.6, 2.7673, 2.7335, 2.5662, 2.4327])
    #
    # test_onunit('sON_TF1', [5.2, 5.3673, 5.3335, 5.1662, 5.0327])
    # test_onunit('sON_TF2', [3.5333, 3.7006, 3.6668, 3.4995, 3.366])
    # test_onunit('sON_TF4', [8.25, 8.4173, 8.3835, 8.2162, 8.0827])
    # test_onunit('sON_TF8', [1.7667, 1.934, 1.9001, 1.7328, 1.5994])
    # test_onunit('sON_TF15', [10.88, 11.0473, 11.0135, 10.8462, 10.7127])
    #
    test_sONsOFF('sONsOFF_001', [4.0, 5.6769, 5.2724, 3.5743, 2.3019])

    # test_sONtOFF('sONtOFF_001', [5.5, 5.5919, 6.5667, 6.47, 5.4033])

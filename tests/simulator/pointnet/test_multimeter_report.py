import pytest
from .conftest import *

try:
    from bmtk.simulator.pointnet.modules.multimeter_reporter import MultimeterMod
except ImportError as ie:
    nest_installed = False


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
@pytest.mark.parametrize('dt', [0.001, 1.0, 5.0, 10.0])
def test_multimeter_dt(dt):
    # Check that a multimeter can be created using large dt values
    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1', batched=True))
    sim = pointnet.PointSimulator(net, dt=dt)
    net.build_nodes()

    mod = MultimeterMod(
        tmp_dir='.',
        file_name='test',
        variable_name=['V_m'],
        cells={'population': 'V1', 'node_id': [0, 1, 2, 3]},
        tstart=0.0,
        tstop=100.0
    )
    mod.initialize(sim)



if __name__ == '__main__':
    test_multimeter_dt(dt=10.0)

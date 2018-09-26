import pytest
import os
import json

import popnet_virtual_files as pvf
from bmtk.simulator import popnet


@pytest.mark.skip()
def test_add_nodes():
    nodes = pvf.NodesFile(N=100)

    net = popnet.PopNetwork()
    net.add_component('models_dir', '.')
    with open('exc_dynamics.json', 'w') as fp:
        json.dump({'tau_m': 0.1}, fp)

    with open('inh_dynamics.json', 'w') as fp:
        json.dump({'tau_m': 0.2}, fp)

    net.add_nodes(nodes)
    assert(net.networks == [nodes.name])
    assert(len(net.get_internal_nodes()) == 2)
    assert(len(net.get_populations(nodes.name)) == 3)
    assert(net.get_populations(nodes.name))

    pop_e = net.get_population(nodes.name, 101)
    assert (pop_e['ei'] == 'e')
    assert (pop_e.is_internal == True)
    assert (pop_e.pop_id == 101)
    assert (pop_e.tau_m == 0.1)

    pop_i = net.get_population(nodes.name, 102)
    assert (pop_i['ei'] == 'i')
    assert (pop_i.is_internal == True)
    assert (pop_i.tau_m == 0.2)


#test_add_nodes()
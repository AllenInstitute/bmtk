import pytest
import os
import numpy as np

from bmtk.utils.reports import CompartmentReport
from bmtk.utils.reports.compartment import plotting

matplotlib = pytest.importorskip('matplotlib')


@pytest.fixture()
def compartment_report():
    cpath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cpath, 'compartment_files/multi_population_report.h5')


@pytest.mark.parametrize('node_ids,average,node_groups', [
    (None, False, None),
    (range(5), False, None),
    ([1, 2, 3], True, None),
    (None, True, None),
    (None, True, [{'node_ids': np.arange(1, 3), 'label': 'some', 'c': 'k'}]),
    (None, False, [{'node_ids': [0], 'label': 'low'},
                   {'node_ids': np.array([1, 2]), 'label': 'mid'},
                   {'node_ids': range(3, 5), 'label': 'high'}])
])
def test_plot_traces(compartment_report, node_ids, average, node_groups):
    fig = plotting.plot_traces(
        report=compartment_report,
        population='v1',
        node_groups=node_groups,
        show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))


if __name__ == '__main__':
    test_plot_traces(compartment_report(), node_ids=None, average=False,
                     node_groups=[{'node_ids': np.arange(1, 3), 'label': 'some', 'c': 'k'}])

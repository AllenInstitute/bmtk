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


@pytest.mark.parametrize('node_ids,population', [
    (1, 'v1'),
    (3, 'v1')
])
def test_plot_single_trace(compartment_report, node_ids, population):
    fig = plotting.plot_traces(
        node_ids=node_ids,
        report=compartment_report,
        population=population,
        show=False
    )
    assert(isinstance(fig, matplotlib.figure.Figure))


def test_plot_invalid_id(compartment_report):
    with pytest.raises(KeyError):
        fig = plotting.plot_traces(
            node_ids=1000,
            report=compartment_report,
            population='v1'
        )


if __name__ == '__main__':
    # test_plot_traces(compartment_report(), node_ids=None, average=False,
    #                  node_groups=[{'node_ids': np.arange(1, 3), 'label': 'some', 'c': 'k'}])
    test_plot_single_trace(
        compartment_report(),
        node_ids=3,
        population='v1'
    )

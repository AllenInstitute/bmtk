import pytest
import numpy as np
import tempfile

from bmtk.utils.reports import SpikeTrains
from bmtk.utils.reports.spike_trains import plotting

matplotlib = pytest.importorskip('matplotlib')


matplotlib.rcParams.update({'figure.max_open_warning': 0})  # stop pytest memory warning


@pytest.fixture
def spike_trains():
    st = SpikeTrains(default_population='V1')
    for n in range(0, 20):
        times = np.random.uniform(0.0, 1500.0, 10)
        times = np.sort(times)
        st.add_spikes(node_ids=n, timestamps=times)

    return st


def test_load_spikes_api(spike_trains):
    fig = plotting.plot_raster(spike_trains=spike_trains, show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))


def test_load_spikes_file(spike_trains):
    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    spike_trains.to_sonata(tmpfile.name)

    fig = plotting.plot_raster(spike_trains=tmpfile.name, show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))

@pytest.mark.parametrize('node_groups', [
    None,
    [{'node_ids': np.arange(10, 20), 'label': 'all', 'c': 'k'}],
    [{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
     {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
     {'node_ids': range(16, 22), 'label': 'high'}]
])
@pytest.mark.parametrize('with_histogram', [
    True,
    False
])
def test_plot_raster(spike_trains, node_groups, with_histogram):
    fig = plotting.plot_raster(spike_trains=spike_trains, node_groups=node_groups, with_histogram=with_histogram,
                               show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))
    assert(len(fig.axes) == 2 if with_histogram else 1)



@pytest.mark.parametrize('node_groups', [
    None,
    [{'node_ids': np.arange(10, 20), 'label': 'all', 'c': 'k'}],
    [{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
     {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
     {'node_ids': range(16, 22), 'label': 'high'}]
])
@pytest.mark.parametrize('smoothing', [
    True,
    False,
    None
])
def test_plot_rates(spike_trains, node_groups, smoothing):
    fig = plotting.plot_rates(spike_trains=spike_trains, node_groups=node_groups, smoothing=smoothing, show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))


@pytest.mark.parametrize('node_groups', [
    None,
    [{'node_ids': np.arange(10, 20), 'label': 'all'}],
    [{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
     {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
     {'node_ids': range(16, 22), 'label': 'high'}]
])
def test_plot_rates_boxplot(spike_trains, node_groups):
    fig = plotting.plot_rates(spike_trains=spike_trains, node_groups=node_groups, show=False)
    assert(isinstance(fig, matplotlib.figure.Figure))


def show_plot():
    st = SpikeTrains(default_population='V1')
    for n in range(0, 100):
        n_vals = np.sin(n*np.pi/100)*150 + 10
        times = np.random.uniform(0.0, 1500.0, int(n_vals))
        times = np.sort(times)
        st.add_spikes(node_ids=n, timestamps=times)

    # plotting.plot_raster(spike_trains=st, title='V1 Spikes')
    # plotting.plot_rates(
    #     spike_trains=st,
    #     node_groups=[{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
    #                  {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
    #                  {'node_ids': range(16, 110), 'label': 'high'}],
    #     smoothing=True
    # )

    node_groups = [{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
                   {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
                   {'node_ids': range(16, 110), 'label': 'high'}]

    plotting.plot_rates_boxplot(
        spike_trains=st,
        node_groups=[{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
                     {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
                     {'node_ids': range(16, 110), 'label': 'high'}]
        # node_groups=node_groups
    )
    print(node_groups)


if __name__ == '__main__':
    # test_load_spikes_api()
    # test_load_spikes_file()
    # test_raster_base()
    # test_raster_no_hist()
    # test_raster_node_groups()
    # show_plot()
    #test_plot_rates(spike_trains=spike_trains(), node_groups=[{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low'},
    # {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
    # {'node_ids': range(16, 110), 'label': 'high'}], smoothing=False)

    # test_plot_raster(
    #     spike_trains=spike_trains(),
    #     node_groups=[{'node_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 'label': 'low', 'c': 'k'},
    #                  {'node_ids': np.array([11, 12, 13, 14, 15]), 'label': 'mid'},
    #                  {'node_ids': range(16, 110), 'label': 'high'}],
    #     with_histogram=False)

    test_plot_raster(
        spike_trains=spike_trains(),
        node_groups=[{'node_ids': np.arange(10, 100), 'label': 'all'}],
        with_histogram=False)

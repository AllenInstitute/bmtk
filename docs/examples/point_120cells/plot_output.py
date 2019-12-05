from bmtk.analyzer.visualization.spikes import plot_raster, plot_rates
import matplotlib.pyplot as plt

plt.figure('Raster')
plot_raster('output/spikes.h5', with_histogram=True, with_labels=['cortex'], show_plot=False)
plt.figure('Rates')
plot_rates('output/spikes.h5', show_plot=False)
plt.show()

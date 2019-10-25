from bmtk.analyzer.visualization.spikes import plot_raster, plot_rates
import matplotlib.pyplot as plt

# Raster plot of the v1 spikes.
plt.figure('Raster')
plot_raster('output/spikes.h5', with_histogram=True, with_labels=['v1'], show_plot=False)
plt.figure('Rates')
plot_rates('output/spikes.h5', show_plot=False)
plt.show()

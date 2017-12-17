from bmtk.analyzer import plot_potential, plot_calcium, spikes_table

print spikes_table(config_file='config.json')

plot_potential(config_file='config.json')
plot_calcium(config_file='config.json')
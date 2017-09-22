from bmtk.simulator.mintnet.analysis.StaticGratings import StaticGratings as StaticGratingsAnalysis
from bmtk.analysis.spikes import plot_tuning
import matplotlib.pyplot as plt

sg_analysis = StaticGratingsAnalysis('sg_bob.ic')
print sg_analysis.node_table

plot_tuning(sg_analysis, 's1', 15, save_as='s1_node15.jpg')
#plot_tuning(sg_analysis, 's1', 0, save_as='s1_node0.jpg', show=False)
#plot_tuning(sg_analysis, 's3', 0, save_as='s3_node0.jpg', show=False)
#plt.show()

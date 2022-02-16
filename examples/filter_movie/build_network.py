import numpy as np
from bmtk.builder import NetworkBuilder


field_size = (304, 608)  # size of movie screen (pixels)
cell_grid = (5, 5)   # place cells in a grid layout of NxM
xs, ys = np.meshgrid(np.linspace(0, field_size[0], num=cell_grid[0]), np.linspace(0, field_size[1], num=cell_grid[1]))

lgn_net = NetworkBuilder('lgn')
lgn_net.add_nodes(
    N=cell_grid[0]*cell_grid[1],
    ei='e',
    model_type='virtual',
    model_template='lgnmodel:LGNOnOFFCell',
    dynamics_params='lgn_on_off_model.json',
    sigma_on=(2.0, 2.0),
    sigma_off=(4.0, 4.0),
    x=xs.flatten(),
    y=ys.flatten()
)

lgn_net.save_nodes(output_dir='network')
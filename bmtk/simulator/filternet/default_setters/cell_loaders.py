from bmtk.simulator.filternet.pyfunction_cache import add_cell_processor


def default_cell_loader(node):
    print node.model_template
    print 'DEFAULT'
    exit()

add_cell_processor(default_cell_loader, 'default', overwrite=False)

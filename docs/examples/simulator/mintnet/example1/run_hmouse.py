from bmtk.simulator.mintnet.hmax.hmax import hmax
from bmtk.simulator.utils.stimulus.StaticGratings import StaticGratings

# train nodes on input images
hm = hmax.load('hmouse/config_hmouse.json', name='hmouse')
hm.train()
hm.generate_output()

# TODO: Check that weights were created

# run layer against a static grating and gather the information
sg_bob = StaticGratings.with_brain_observatory_stimulus(num_trials=1)
exemplar_node_table = hm.get_exemplar_node_table()
hm.run_stimulus(sg_bob, node_table=exemplar_node_table, output_file='sg_bob1')

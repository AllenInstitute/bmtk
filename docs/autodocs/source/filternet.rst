FilterNet
=========
FilterNet will simulate the effects of visual stimuli onto a receptive field. It uses LGNModel simulator as a backend, which
uses neural-filters to simulate firing rates and spike-trains over a given time-course and stimuli.


Features
--------
* Supports a number of stimuli inputs including:
 * Static and moving grating
 * Full field flashes
 * Static images
 * Short movies.


Installation
------------
Filter supports both Python 2.7 or Python 3.6+, see our `Installation instructions <installation>`_.



Documentation and Tutorials
---------------------------
For more information about FilterNet and LGNModel in particular, please contact the main developers Ram Iyer (rami at alleninstitute dot org)
and Yazan Billeh (yazanb at alleninstitute dot org).


Examples
--------
The AllenInstitute/bmtk repo contains a number of FilterNet examples, many with pre-built networks and can be immediately ran. These
tutorials will have the folder prefix *filter_* and to run them in the command-line simply call::

  $ python run_filternet.py config.json


Current examples
++++++++++++++++
* `filter_graitings <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/filter_graitings>`_ - An example of a 2 second static grating stimuli on the LGN receptive field.

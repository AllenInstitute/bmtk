.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: User Guide

    Overview of BMTK <self>
    builder
    simulators_guide
    analyzer


##########
User Guide
##########

******************************************
Overview of BMTK Workflow and Architecture
******************************************



The Brain Modeling Toolkit (BMTK) is a python-based software package for building, simulating and analyzing large-scale
neural network models. It supports the building and simulation of models of varying levels-of-resolution; from
multi-compartment biophysically-detailed networks, to point-neuron models, to filter-based models, and even
population-level firing rate models.

.. figure:: _static/images/levels_of_resolution_noml.png
   :scale: 70%

The BMTK is not itself a simulator and utilizes existing simulators, like NEURON and NEST, to run different types of
models. What BMTK does provide is a streamlined workflow to build, analyze, and store models efficiently:

.. figure:: _static/images/bmtk_architecture.jpg
   :scale: 45%

The BMTK Workflow and architecture
----------------------------------
BMTK can readily scale to run models of single neurons, and even single compartments, for all different types of
neuronal networks. However BMTK was designed for very-large, highly optimized mammalian cortical network models.
Generating the connectivity matrix could take hours or even days to complete. We can then test these optimized base-line 
models against a large variety of conditions and stimuli and directly compare with existing in-vivo
recordings (see Allen Brain Observatory).

.. figure:: _static/images/bmtk_workflow.jpg
   :scale: 100%

Unlike other simulators, BMTK separates the process of building, simulating, and analyzing the results. First a fully
instantiated base-line version of the model is built and saved to a file so that each time a simulation runs it takes only
a small fraction of the time to instantiate the simulation. Results are also saved automatically. BMTK and
the format it uses (SONATA, see below) makes it easy to dynamically adjust cell and synaptic parameters so that multiple
iterations of a simulation can be done as fast as possible.

As such BMTK can be broken into three major components:

* The Network Builder [:py:mod:`bmtk.builder`] - Used to build network models
* The Simulation Engines [:py:mod:`bmtk.simulator`] - Interfaces for simulating the network
* Analysis and Visualization Tools [:py:mod:`bmtk.analyzer`] - Python functions for analyzing and visualizing the
   network and simulation results

.. figure:: _static/images/bmtk_architecture.jpg
   :scale: 45%

The components can function as one workflow or separately by themselves. BMTK utilizes the SONATA Data format (see next section)
to allow sharing of large-scale network data. As such, it is possible to use the Network Builder to build the model but
another tool to run the model. Or if another model has been built by someone else and saved in SONATA format, BMTK will
be able to simulate it.


.. _cards-clickable:

Getting Started
===============
.. grid:: 1 1 3 3
    :gutter: 1

    .. grid-item-card::  Building Networks
        :link: builder.rst
        :img-bottom: _static/images/builder_complete_network.jpg

        *NetworkBuilder* 


    .. grid-item-card:: Network Simulations
        :link: simulators_guide.rst
        :img-bottom: _static/images/brunel_comparisons.jpg

        *Setting up environments, BioNet, PointNet, PopNet* 

       
    .. grid-item-card:: Analyzing Networks and Simulations
        :link: analyzer.rst
        :img-bottom: _static/images/raster_120cells_orig.png

        *Plotting simulation results and statistics*      




Related Resources
=================

.. grid:: 1 1 3 3
    :gutter: 1

    .. grid-item-card:: Workshops and Tutorials
        :link: https://github.com/AllenInstitute/bmtk-workshop
        :img-bottom: _static/images/jupyter_notebook_generic.png

        *Link to workshops and other tutorials*

    .. grid-item-card:: Network and Simulation visualization
        :link: https://www.ks.uiuc.edu/Research/vnd/
        :img-bottom: _static/images/mousev1_vnd.png 

        *VND*


    .. grid-item-card:: The SONATA data format
        :link: https://github.com/AllenInstitute/sonata
        :img-bottom: _static/images/segmentation_indexing.jpg 

        *SONATA format guide and API*

    .. grid-item-card:: Allen Institute Mouse Models
        :link: https://portal.brain-map.org/explore/models
        :img-bottom: https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/e3/fe/e3fedb92-d998-42c7-9799-3f19160cc2ca/computational-modeling-theory-140x140-72dpi.png

        *Link to brain-map portal*



Core Guides
===========
.. toctree::
   :hidden:

   About BMTK <self>
   news_and_events
   contact_us
   installation
   user_guide
   tutorials
   how_to_cite
   developers_guide


################################
BMTK: The Brain Modeling Toolkit
################################

.. figure:: _static/images/mousev1_banner_compressed.png


.. raw:: html

   <a href="https://secure2.convio.net/allins/site/SPageServer/?pagename=modeling_tools" style="float: right;">
      <button>Subscribe to our newsletter</button>
   </a>


About BMTK
==========

.. card::

   The Brain Modeling Toolkit (BMTK) is a open-source software package for modeling and simulation of large-scale neural 
   network models. It Supports a range of modeling levels of resolution: multi-compartment biophysically detailed, 
   point-neuron, and population-level firing rate models. 
   
   BMTK supports the full workflow for developing biologically realistic models of the brain networks; from building 
   network models from scratch, to running parallelized simulations, to doing perturbation analysis. It offers:

   .. grid:: 2

      .. grid-item:: 

         * An interface (API) for building models and running simulations, unified across levels of resolution
         
         * A framework to easily share models and expand upon existing models
         
         * A simple network simulation setup with little-to-no programming necessary

         * Adaptability that allows more advanced users to completely change how networks are instantiated and simulated
          
         * Auotmatic parallelization

         * Support for simulations ranging from single cells to networks with millions of cells and billions of synapses

         * Functionality to programmatically adjust cell and synaptic properties on-the-fly during simulations

         * A suite of functions for analyzing and visualizing network structure and simulations results

         And much more!

   
      .. grid-item::

         .. figure:: _static/images/v1_dg_500x333.gif


   **BMTK Workflow**

   BMTK separates the process of building, simulating, and analyzing the results, thanks to modular organization and the
   data format it uses (`SONATA <https://github.com/AllenInstitute/sonata>`__, see below). BMTK constructs a fully instantiated model and saves it in SONATA files. 
   Subsequently, running a simulation involves loading SONATA files, without the need to re-build the model.

   On the other hand, users can easily adjust cell and synaptic parameters in SONATA files, enabling faster iterations
   of simulations. Models in SONATA format can be constructed, simulated, or analyzed using not only BMTK, but also a 
   variety of other tools that support this format.

   BMTK consists of three major components: `the network builder <builder.hmtl>`__, the 
   `simulation engines <simulators_guide.html>`__, and the 
   `analysis and visualization tools <analyzer.html>`__. The components can be used in one workflow or separately.

   .. raw:: html

      <div style="position: relative;">
         <img src="_static/images/bmtk-workflow-ver2.png" style="width: 100%; height: auto;">
         <a href="builder.html"><div style="position: absolute; left: 0%; top: 22%; width: 20%; height: 13%; background-color: rgba(0, 0, 0, .0);"></div></a>
         <a href="simulators_guide.html"><div style="position: absolute; left: 38%; top: 22%; width: 23%; height: 13%; background-color: rgba(0, 0, 0, 0);"></div></a>
         <a href="analyzer.html"><div style="position: absolute; left: 78%; top: 22%; width: 22%; height: 13%; background-color: rgba(0, 0, 0, .0);"></div></a>
         
         <a href="bionet.html"><div style="position: absolute; left: 22%; top: 48%; width: 13%; height: 52%; background-color: rgba(0, 0, 0, .0);"></div></a>
         <a href="pointnet.html"><div style="position: absolute; left: 36%; top: 48%; width: 13%; height: 52%; background-color: rgba(0, 0, 0, .0);"></div></a>
         <a href="filternet.html"><div style="position: absolute; left: 50%; top: 48%; width: 13%; height: 52%; background-color: rgba(0, 0, 0, .0);"></div></a>
         <a href="popnet.html"><div style="position: absolute; left: 64%; top: 48%; width: 13%; height: 52%; background-color: rgba(0, 0, 0, .0);"></div></a>
      </div>


Further Resources
=================

.. grid:: 1 1 2 2
   :gutter: 1

   .. grid-item-card:: `User Guide <user_guide.html>`__

      For detailed information about using BMTK and all the available features please see our `User Guide <user_guide>`__ page.

   .. grid-item-card:: `Tutorials and Examples <tutorials.html>`__

      For a list of workable tutorials and example networks please see our `Tutorials and Examples <tutorials.html>`_ page.      

   .. grid-item-card:: Allen Institute Brain Map Portal

      For examples of how we use BMTK and related tools to build realistic models at the Allen Institute, please see our 
      `Computational Modeling & Theory page at the Allen Brain Map Portal <https://portal.brain-map.org/explore/models>`__.

   .. grid-item-card:: `Contact Us <contact_us.html>`__ 

      If you have questions, issues, or requests for BMTK developers and/or the scientists that use the tool in their 
      own modeling work, please feel free to reach out to us directly. Our `Contact Us <contact_us.html>`__ page contains 
      a number of ways to reach out to our team at the Allen Institute.


Related Tools
-------------

.. grid:: 1

   .. grid-item-card:: SONATA Data Formats

      SONATA is a cross-platform data format for storing and exchaning large scale networks and simulation results. For 
      more inofmration see the `SONATA github page <https://github.com/AllenInstitute/sonata>`_ and the 
      `SONATA Paper <how_to_cite.html>`__.

   .. grid-item-card:: Visual Neuronal Dynamics (VND)

      `Visual Neuronal Dynamics <https://www.ks.uiuc.edu/Research/vnd/>`_ is a software package for 3D visualization of neuronal network models. VND can be used to check and inspect models, such as those created by BMTK, as well as to visualize the activity output. Images and movies made in VND can also be used to showcase or schematize models for presentations and publications. VND is developed in a collaboration between researchers at the University of Illinois at Urbana-Champaign and Allen Institute.  


Acknowledgements
================

.. card::

   `How to cite <how_to_cite>`__ BMTK and related tools.

.. card:: 

   See our `Contributors Page <contributors.html>`__ for a list of the people who have helped with the development and growth of BMTK.

.. card:: 

   We wish to thank the Allen Institute founder, Paul G. Allen, for his vision, encouragement, and support.


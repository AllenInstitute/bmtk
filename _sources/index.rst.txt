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

   <a href="https://signup.e2ma.net/signup/2010845/1976001/" style="float: right; font-size: x-large; font-weight: bold;">
      <button>Subscribe to our newsletter</button>
   </a>


About BMTK
==========

.. card::

   The Brain Modeling Toolkit (BMTK) is an **open-source** software package for modeling and simulating large-scale
   neural network models. It supports a range of modeling resolutions, including **multi-compartment, biophysically 
   detailed** models, **point-neuron** models, and **population-level firing rate** models.
   
   BMTK provides a **full workflow** for developing biologically realistic brain network models—from **building 
   networks from scratch**, to running **parallelized simulations**, to conducting **perturbation analyses**. Its 
   features include:

   .. grid:: 2

      .. grid-item:: 

         * A **unified interface (API)** for building and running simulations across levels of resolution
         
         * A **flexible framework** for sharing models and expanding upon existing ones
         
         * A **simple simulation setup** requiring little-to-no programming

         * **Adaptability** that allows advanced users to fully customize how networks are instantiated and simulated
          
         * **Automatic parallelization**

         * Support for simulations ranging from single cells to networks with millions of cells and billions of synapses

         * **On-the-fly adjustments** of cell and synaptic properties during simulations

         * **A suite of functions** for analyzing and visualizing network structure and simulation results

         And much more!

   
      .. grid-item::

         .. figure:: _static/images/v1_dg_500x333.gif
            :scale: 160%

            `Simulation of the mouse primary visual cortex <https://portal.brain-map.org/explore/models/mv1-all-layers>`__, prepared and carried out using BMTK.


   **BMTK Workflow**

   BMTK uses a **modular organization** and the **SONATA** data format (`SONATA <https://github.com/AllenInstitute/sonata>`__, 
   see below) to separate the processes of **building**, **simulating**, and **analyzing** brain network models. BMTK
   constructs a fully instantiated model and saves it in SONATA files. Then, running a simulation simply involves
   loading those SONATA files, with no need to rebuild the model each time.

   Users can also adjust cell and synaptic parameters in the SONATA files to iterate more quickly on simulations. 
   Models in SONATA format can be constructed, simulated, or analyzed not only with BMTK but also with other tools 
   that support the format.

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


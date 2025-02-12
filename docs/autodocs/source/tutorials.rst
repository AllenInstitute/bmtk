######################
Tutorials and Examples
######################

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: A
    
    Builder: Using the Network Builder <tutorials/NetworkBuilder_Intro>
    BioNet: Single cell with current injection <tutorials/tutorial_01_single_cell_clamped>
    BioNet: Single cell with with synaptic input <tutorials/tutorial_02_single_cell_syn>
    BioNet: Multiple Nodes with single cell-type <tutorials/tutorial_03_single_pop>
    BioNet: Heterogeneous network <tutorials/tutorial_04_multi_pop>
    PointNet: Point-neuron modeling <tutorials/tutorial_05_pointnet_modeling>
    FilterNet: Full-field flashing movie <tutorials/tutorial_07_filter_models>
    Auditory FilterNet: Generating stimuli from auditory input <tutorials/auditory_filternet>
    PopNet: Population-based firing rate models <tutorials/tutorial_06_population_modeling>
    ad_tutorials
    examples


Basic Usage
===========

.. grid:: 1 1 3 3
    :gutter: 1

    .. grid-item-card::  Overview 

        A high level look at the BMTK workflow, SONATA data-format, and how to 
        create an environment for network modeling and simulation


    .. grid-item-card:: Building Networks models with the BMTK NetworkBuilder 
        :link: tutorial_NetworkBuilder_Intro.html

        Using the BMTK NetworkBuilder to create SONATA based network models for use in simulation and analysis

       
    .. grid-item-card:: Simulating biologically detailed networks

        How to use BMTK BioNet to run simulations of networks of biophyscially realistic compartmental
        cell models.        


    .. grid-item-card:: Simulating point-neuron networks with PointNet

        How to use BMTK PointNet for running simulation of single point-neuron models.


    .. grid-item-card:: Generating releastic sensory stimuli with FilterNet

        Use BMTK FilterNet to convert stimuli into a series of spikes for analysis and network 
        stimuli.


    .. grid-item-card:: Analyzing simulation results

        How to use bmtk to fetch data from simulation results, plot and get statistics.




More Features
=============

.. grid:: 1 1 3 3
    :gutter: 1

    .. grid-item-card:: Parallelization and Threading Options 

        Options for running large-scale simulations taking advantage of multiple cores and Threading
        with HPC

    .. grid-item-card:: Advanced Stimulus for simulations 

        More options for input stimulus to simulations


    .. grid-item-card:: Advanced simulation recordings 

        Includes voltage. LFP, synapse recording

    .. grid-item-card:: Customized Python  

        Creating modules

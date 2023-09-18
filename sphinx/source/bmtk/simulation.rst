Simulating the neural response with BMTK
========================================

In order to simulate the neural activation of the tissue in response to the extracellular potentials, we need two things:

-  The extracellular potentials which were calculated in COMSOL.
-  A computational model of the neural tissue in BMTK.

Then, we can use the comsol module in BMTK’s BioNet to import the COMSOL output and simulate the behaviour of the neural network in response to the imposed extracellular potentials.

Building a network
------------------

Thanks to the work of `Billeh et al. <https://doi.org/10.1016/j.neuron.2020.01.040>`__, a model of the mouse V1 already exists.
However, it is possible to adapt certain parameters (e.g. the size or shape of the patch of tissue) to fit the specific needs of an experiment. 
Changing neuron models or synaptic connections is also possible, but obviously it is a lot more complex and might disrupt the proper functioning of the model.

In the context of ReVision’s computational experiments, we will create multiple network models –each with a different random seed– in order to perform the same experiment multiple times on different virtual ‘animals’. 
While the neuron types are constrained to a certain layer of V1 and the general connection rules between and within layers are set, the random seed determines the positions of the neurons within their respective layer, as well as the exact synaptic connections that are made between individual neurons.

Building a network uses several components/scripts that are found in the v1 folder. 
- v1/build_files/ 
- v1/components/ 
- v1/build_network.py
Simple model tweaks can probably be made here.

Calling ``build_network.py`` in the terminal or with a bash/job script (with a few optional arguments) will actually build the network including any possible changes you might have made.

::

   $ python build_network.py -o [output] --fraction [fraction] --rng-seed [seed]

-  [output] - Output directory. Defaults to networks_rebuilt/network.
-  [fraction] - Fraction of the total neurons the model should include. Defaults to 1.
-  [seed] - Random seed to build the network, changing this will allow the creation of different virtual ‘animals’. Defaults to 100.

For parallel computing, add ``mpirun -np [np]`` before the command above. 

- [np] - number of cores to use.

Calling ``build_network.py`` multiple times with different random seeds (make sure to also set a different output directory!) will instantiate several networks that represent the different virtual ‘animals’::

    mpirun -np 12 python build_network.py --fraction 1 -o networks_25/network0 --rng-seed 100
    mpirun -np 12 python build_network.py --fraction 1 -o networks_25/network1 --rng-seed 101
    mpirun -np 12 python build_network.py --fraction 1 -o networks_25/network2 --rng-seed 102

Generating waveform.csv (only for stationary COMSOL study/studies)
------------------------------------------------------------------

Generating the ``waveform.csv`` file is done with :py:mod:`waveform`. 
The main class is called :py:class:`toolbox.waveform.CreateWaveform` and constructs the waveform from a piecewise description.

.. autoclass:: toolbox.waveform.CreateWaveform
  :no-index:

There is an additional function :py:class:`toolbox.waveform.CreateBlockWaveform` that can be used to more easily construct a waveform consisting of rectangular pulses.

.. autofunction:: toolbox.waveform.CreateBlockWaveform
  :no-index:

Running :py:mod:`waveform` will execute the lines below::

   if __name__ == '__main__':
       '''
       If you run the file itself instead of calling if from another file, this part will run.
       
       '''

Running a simulation
--------------------

Once you have built one or several networks, you can run simulations
with the previously built network(s). This also requires extracellular
potentials that were calculated in COMSOL.
Depending on the stimulation parameters, the COMSOL output should be
either stationary or time-dependent.

Depending on the type of COMSOL study/studies you chose, configuring the
comsol input for BMTK in the config.json file, will look slightly
different.

One time-dependent study
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: json

       "Extracellular_Stim": {
           "input_type": "lfp",
           "node_set": "all",
           "module": "comsol",
           "comsol_files": "$STIM_DIR/comsol.txt",
           "amplitudes": 1,
       }

One stationary study
~~~~~~~~~~~~~~~~~~~~

.. code:: json

       "Extracellular_Stim": {
           "input_type": "lfp",
           "node_set": "all",
           "module": "comsol",
           "comsol_files": "$STIM_DIR/comsol.txt",
           "waveforms": "$STIM_DIR/waveform.csv",
           "amplitudes": 1,
       }

Multiple stationary studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: json

       "Extracellular_Stim": {
           "input_type": "lfp",
           "node_set": "all",
           "module": "comsol",
           "comsol_files": ["$STIM_DIR/comsol_1.txt", "$STIM_DIR/comsol_2.txt", "..."],
           "waveforms": ["$STIM_DIR/waveform_1.csv", "$STIM_DIR/waveform_2.csv", "..."],
           "amplitudes": [1, 1, ],
       }

You should probably never change the first three arguments: 

- input_type: Has to be “lfp”. 
- node_set: Used to filter which cells receive the input, but here it probably does not make sense to use anything besides “all”. 
- module: Has to be “comsol”.

You should probably change the other arguments:

- comsol_files 

  - One study: (path) “/path/to/comsol.txt”
  - Multiple stationary studies: (list) List of paths.

- waveforms

  - One time dependent study: Remove ``"waveforms": ...`` line from config.json.
  - One stationary study: (path) “/path/to/waveform.csv”. 
  - Multiple stationary studies: (list) List of paths to the different waveform.csv files. 

- amplitudes: Scaling factor for waveform. 

  - E.g. if the amplitudes in waveform.csv are normalised to [-1;1], this can be used to set the current amplitude. Defaults to 1. 
  - One study: (float)
  - Multiple studies: (list or float) List of waveform amplitudes. Float can be used if all amplitudes are identical.
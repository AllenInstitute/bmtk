[Back to ToC](/docs/manual/README.md)

# Simulating the neural response with BMTK

In order to simulate the neural activation of the tissue in response to the extracellular potentials, we need two things:

- The extracellular potentials which were [calculated in COMSOL](../comsol/solution.md).
- A computational model of the neural tissue in BMTK.

Then, we can use the [comsol](/examples/comsol/README.md) module in BMTK's BioNet to import the COMSOL output and simulate the behaviour of the neural network in response to the imposed extracellular potentials.

## Building a network

Thanks to the work of [Billeh et al.](https://doi.org/10.1016/j.neuron.2020.01.040), a model of the mouse V1 already exists. However, it is possible to adapt certain parameters (e.g. the size or shape of the patch of tissue) to fit the specific needs of an experiment. Changing neuron models or synaptic connections is also possible, but obviously it is a lot more complex and might disrupt the proper functioning of the model.

In the context of ReVision's computational experiments, we will create multiple network models &ndash;each with a different random seed&ndash; in order to perform the same experiment multiple times on different virtual 'animals'. While the neuron types are constrained to a certain layer of V1 and the general connection rules between and within layers are set, the random seed determines the positions of the neurons within their respective layer, as well as the exact synaptic connections that are made between individual neurons.  

Building a network uses several components/scripts that are found in the v1 folder. 
- v1/build_files/ 
- v1/components/
- v1/build_network.py - Simple model tweaks can probably be made here.


Calling `build_network.py` in the terminal or with a bash/job script (with a few optional arguments) will actually build the network including any possible changes you might have made. 
```
$ python build_network.py -o [output] --fraction [fraction] --rng-seed [seed]
```
- [output] - Output directory. Defaults to networks_rebuilt/network.
- [fraction] - Fraction of the total neurons the model should include. Defaults to 1.
- [seed] - Random seed to build the network, changing this will allow the creation of different virtual 'animals'. Defaults to 100.

For parallel computing, add ```mpirun -np [np]``` before the command above.
- [np] - number of cores to use.

Calling `build_network.py` multiple times with different random seeds (make sure to also set a different output directory!) will instantiate several networks that represent the different virtual 'animals'.

An example of a bash script calling this function can be found [here](/v1/build.sh). 

## Generating waveform.csv

### Only required in combination with Stationary studies. Skip this section if you used a Time Dependent study in COMSOL.

Generating the `waveform.csv` file is done with [/v1/components/waveform.py](/v1/components/waveform.py). The main class is called `CreateWaveform()` and constructs the waveform from a piecewise description.

```python 
class CreateWaveform(
    """
    piecewise - (N_pieces x 2 array) Piecewise description of the waveform. 
        Each row in the form [t_stop, lambda t:func(t)]
        The lambda expression is valid between t_start and t_stop.
        t_start=0 for the first segment, otherwise t_stop of the previous segment. 
    amplitude - (float) If specified, waveform is rescaled to this amplitude. Defaults to None.
    dt - (float) Time step between two points in ms. Defaults to 0.025.
    path - (str) /path/to/save/waveform.csv. Defaults to None, in which case the waveform is not saved.
    plot - (bool) If True, plot the waveform. Defaults to False.
    """
)
```

There is an additional function `CreateBlockWaveform()` that can be used to more easily construct a waveform consisting of rectangular pulses.

```python
def CreateBlockWaveform(
    """
    n_pulses - (int) Number of pulses that the waveform should comprise. 
    Lambda expressions used to define the pulse parameters as a function of the pulse.
        phase_1_expr - (lambda) Duration of the first phase of the pulse.
        amp_1_expr - (lambda) Amplitude of the first phase of the pulse.
        T_1_expr - (lambda) Time between the end of the first phase and the start of the second phase.
        phase_2_expr - (lambda) Duration of the second phase of the pulse.
        amp_2_expr - (lambda) Amplitude of the second phase of the pulse.
        T_2_expr - (lambda) Time between the end of one pulse and the start of the next.
    save_name - (str) Name of the file (saved in /v1/components/stimulations/). Defaults to None, in which case the waveform is not saved.
    """
)
```

Running `waveform.py` will execute the lines below

```python
if __name__ == '__main__':
    '''
    If you run the file itself instead of calling if from another file, this part will run.
    
    '''
```

## Running a simulation

Once you have built one or several networks, you can run simulations with the previously built network(s). This also requires extracellular potentials that were [calculated in COMSOL](../comsol/solution.md). Depending on the stimulation parameters, the COMSOL output should be either stationary or time-dependent.

Depending on the type of COMSOL study/studies you chose, configuring the comsol input for BMTK in the config.json file, will look slightly different.


### One time-dependent study

```json
    "Extracellular_Stim": {
        "input_type": "lfp",
        "node_set": "all",
        "module": "comsol",
        "comsol_files": "$STIM_DIR/comsol.txt",
        "amplitudes": 1,
    }
```

### One stationary study
```json
    "Extracellular_Stim": {
        "input_type": "lfp",
        "node_set": "all",
        "module": "comsol",
        "comsol_files": "$STIM_DIR/comsol.txt",
        "waveforms": "$STIM_DIR/waveform.csv",
        "amplitudes": 1,
    }
```

### Multiple stationary studies
```json
    "Extracellular_Stim": {
        "input_type": "lfp",
        "node_set": "all",
        "module": "comsol",
        "comsol_files": ["$STIM_DIR/comsol_1.txt", "$STIM_DIR/comsol_2.txt", ...],
        "waveforms": ["$STIM_DIR/waveform_1.csv", "$STIM_DIR/waveform_2.csv", ...],
        "amplitudes": [1, 1, ...],
    }
```
You should probably never change the first three arguments:
- input_type - Has to be "lfp".
- node_set - Used to filter which cells receive the input, but here it probably does not make sense to use anything besides "all".
- module - Has to be "comsol".

You should probably change the other arguments:
- comsol_files -
    - One study: (str) "/path/to/comsol.txt"
    - Multiple stationary studies: (list) List of paths to the different comsol.txt files.
- waveforms - 
    - One time dependent study: Remove `"waveforms": ...` line from config.json.
    - One stationary study: (str) "/path/to/waveform.csv".
    - Multiple stationary studies: (list) List of paths to the different waveform.csv files.
- amplitudes - Scaling factor for waveform. E.g. if the amplitudes in waveform.csv are normalised to [-1;1], this can be used to set the current amplitude. Defaults to 1. 
    - One study: (float)
    - Multiple studies: (list or float) List of waveform amplitudes. Float can be used if all amplitudes are identical.

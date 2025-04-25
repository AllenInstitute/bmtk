# %% test run of the SONATA version of the network

# This is merely trying to check the validity of the files, and not specifying how
# the files should be actually read.
# In particular, in this example, I'm assuming one node per population, which is not
# generally the case.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

# load the input files
l4e_input = np.load("input/l4e_rates.npy")
bkg_input = np.load("input/bkg_rates.npy")
init_state = np.load("input/initial_state.npy")
ext_inputs = np.concatenate([l4e_input, bkg_input], axis=1)


# load the nodes
network_dir = "network.orig"
pops = ["l23", "l4e", "bkg"]

nodes = []
for pop in pops:
    filename = f"{network_dir}/{pop}_node_types.csv"
    df = pd.read_csv(filename, sep=" ")
    nodes.append(df)

# concatenate them

nodes_recurrent = nodes[0]
n_neu_recurrent = len(nodes_recurrent)  # number of recurrent neurons

nodes_all = pd.concat(nodes, ignore_index=True)
n_neu_total = len(nodes_all)  # total number of neurons

# extract necessary elements for simulation
scales = nodes_recurrent["scaling_coef"].values
input_offset = nodes_recurrent["input_offset"].values
exponents = nodes_recurrent["exponent"].values
decay_constants = nodes_recurrent["decay_const"].values
dt = 1.0  # ms  this should come from the config file


# %% load the edges
edges = []
for pop in pops:
    filename = f"{network_dir}/{pop}_l23_edge_types.csv"
    df = pd.read_csv(filename, sep=" ")
    edges.append(df)

# construct the connectivity matrix
edges_all = pd.concat(edges, ignore_index=True)
# transpose to multiply to state vector
mat = edges_all["syn_weight"].values.reshape([6, 4]).T


# %% define the simulation
# from numba import njit


@njit
def relu(array):  # non destructive; a litte slower, but safer
    return array * (array > 0)  # still faster than np.maximum()


@njit
def relu2(array):
    # if the element is negative, set it to zero using loop
    # destructive method (alters the original array), but faster than the above one.
    for i in range(len(array)):
        if array[i] < 0:
            array[i] = 0
    return array


@njit
def step(state, mat, scales, exponents, decay_constants, dt):
    """Execute One step of the SSN model
    Arguments:
        state {np.ndarray} -- current state (firing rates) of the network including the
                              external inputs
        mat {np.ndarray} -- connectivity matrix  (N, N + M)
        scales {np.ndarray} -- input scaling coefficients
        exponents {np.ndarray} -- exponents (alpha)
        decay_constants {np.ndarray} -- decay constants (tau)
        dt {float} -- time step size

        N and M are the number of recurrent and external neurons, respectively
    Returns:
        np.ndarray -- updated state of the network (only recurrent neurons)
    """
    input = relu2(np.dot(mat, state) * scales) ** exponents
    n_neu_recurrent = mat.shape[0]
    recurrent_state = state[:n_neu_recurrent]
    dr = (-recurrent_state + input) / decay_constants * dt
    return relu2(recurrent_state + dr)


@njit
def simulate(init_state, ext_inputs, mat, scales, exponents, decay_constants, dt):
    """Simulate the network
    Arguments:
        init_state {np.ndarray} -- initial state of the network (only recurrent neurons)
        ext_inputs {np.ndarray} -- external inputs (n_step, n_pops) matrix
        mat {np.ndarray} -- connectivity matrix  (N, N + M)
        scales {np.ndarray} -- input scaling coefficients
        exponents {np.ndarray} -- exponents (alpha)
        decay_constants {np.ndarray} -- decay constants (tau)
        dt {float} -- time step size
    Returns:
        np.ndarray -- simulation results (n_step, n_neu_recurrent)
    """
    # state is initial state
    # ext_inputs is profile of external inputs. (n_step, n_pops) matrix
    n_steps = ext_inputs.shape[0]
    n_neu_recurrent = len(init_state)
    n_neu_total = n_neu_recurrent + ext_inputs.shape[1]

    # define the results including the external inputs.
    results = np.zeros((n_steps, n_neu_total))
    results[:, n_neu_recurrent:] = ext_inputs
    # fill in the first time step with the initial state
    results[0, :n_neu_recurrent] = init_state


    for t in range(n_steps - 1):
        results[t + 1, :n_neu_recurrent] = step(
            results[t, :],
            mat,
            scales,
            exponents,
            decay_constants,
            dt,
        )
        # state = step(state, mat, scales, exponents, decay_constants, dt)
        # results[t + 1, :n_neu_recurrent] = state

    return results[:, :n_neu_recurrent]  # return only the recurrent neurons


print(f'init_state: {init_state}')
print(f'ext_inputs: {ext_inputs}')
print(f'mat: {mat}')
print(f'scales: {scales}')
print(f'exponents: {exponents}')
print(f'decay_constants: {decay_constants}')
print(f'dt: {dt}')
# , scales, exponents, decay_constants, dt

# print(scales)
# print(init_state)
# exit()

# run the simulation
result = simulate(init_state, ext_inputs, mat, scales, exponents, decay_constants, dt)
# %timeit simulate(init_state, ext_inputs, mat, scales, exponents, decay_constants, dt)
# 5.87 ms ± 10.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

print(result)

# %%
fig, ax = plt.subplots(2, 1)
ax[0].plot(result[:, :])
ax[0].legend(nodes_recurrent["cell_types"].values)
ax[0].title.set_text("Entire simulation")


# usually, I only use the latter half of the simulation because the first half is
# a settling period, but showing the whole simulation here for demonstration.

ax[1].plot(result[6750:, :])  # if you only need the latter half
ax[1].title.set_text("After settling")


# save the figure
fig.tight_layout()
fig.savefig("test_run.png", dpi=100)

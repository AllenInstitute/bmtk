# %% quick network creation example
import pandas as pd
import h5py as h5
import numpy as np
import model_data

# make an example network for a l2/3 simulation.
# in this simulation, each population type only has one population.
# layer 2/3 (4 cell types) and layer 4 excitatory (1 cell type)
pops = ["l23", "l4e", "bkg"]

# parameters are derived from the pre-run optimization.
# connections are multiplied by the expected influence matrix from the data.
p = pd.read_csv("demo_params.csv", index_col=0, header=None).squeeze("columns")
scale = p["conn_scale"]

decays = p[["tau_e", "tau_p", "tau_s", "tau_v"]].values * 1000  # s to ms
l4e_to_l23 = (
    p[["stim_e", "stim_p", "stim_s", "stim_v"]].values
    * model_data.e4_l23_infl_matrix_mean
)

# divide by scale to be consistent with the new framework
bkg_to_l23 = p[["input_e", "input_p", "input_s", "input_v"]].values / scale
celltypes = ["e", "p", "s", "v"]
# get the relative value from the fit


l23_to_l23 = p[[f"{p1}_to_{p2}" for p1 in celltypes for p2 in celltypes]].values
# then, multiply by the expected influence matrix
print(l23_to_l23)

l23_to_l23 = l23_to_l23 * np.array(model_data.l23_infl_matrix_mean).flatten()


dicts = {
    "l23": {
        "node_type_id": [0, 1, 2, 3],
        "pop_name": ["L23"] * 4,
        "cell_types": ["Exc", "PV", "SST", "VIP"],  # optional
        "model_type": ["rate_population"] * 4,
        "model_template": [None] * 4,
        "scaling_coef": [scale] * 4,
        "input_offset": [0.0] * 4,
        "exponent": [2.0] * 4,
        "decay_const": decays,
    },
    "l4e": {
        "node_type_id": [0],
        "pop_name": ["L4Exc"],
        "model_type": ["virtual"],
        "model_template": [None],
    },
    "bkg": {
        "node_type_id": [0],
        "pop_name": ["BKG"],
        "model_type": ["virtual"],
        "model_template": [None],
    },
}


def create_node_h5(popname, node_type_ids):
    # create an h5 file with one node per node_type_id
    # each node has a unique node_id
    f = h5.File(f"network/{popname}_nodes.h5", "w")
    f.create_group(f"/nodes/{popname}")
    f.create_dataset(f"/nodes/{popname}/node_type_id", data=node_type_ids)
    f.create_dataset(f"/nodes/{popname}/node_id", data=node_type_ids)
    f.close()


# generate and save them as csv files in the network directory
for pname in pops:
    # create the node types csv
    df = pd.DataFrame(dicts[pname])
    df.to_csv(f"network/{pname}_node_types.csv", index=False, sep=" ")
    create_node_h5(pname, dicts[pname]["node_type_id"])


def create_edge_h5(pop1, pop2, edge_type_ids):
    # the edges are also defined only per type, so this will only contain trivial
    # information.
    f = h5.File(f"network/{pop1}_{pop2}_edges.h5", "w")
    f.create_group(f"/edges/{pop1}_{pop2}")
    edge_ids = np.array(range(len(edge_type_ids)))
    source_node_id = np.array(edge_type_ids) // len(dicts[pop2]["node_type_id"])
    target_node_id = np.array(edge_type_ids) % len(dicts[pop2]["node_type_id"])
    f.create_dataset(f"/edges/{pop1}_{pop2}/edge_type_id", data=edge_type_ids)
    f.create_dataset(f"/edges/{pop1}_{pop2}/edge_id", data=edge_ids)
    f.create_dataset(f"/edges/{pop1}_{pop2}/source_node_id", data=source_node_id)
    f.create_dataset(f"/edges/{pop1}_{pop2}/target_node_id", data=target_node_id)
    f.close()


# target is always l23, so we can loop over the sources
weight_dicts = {
    "l23": l23_to_l23,
    "l4e": l4e_to_l23,
    "bkg": bkg_to_l23,
}

target_ids = dicts["l23"]["node_type_id"]  # types id are the same as node ids
for pname in pops:
    source_ids = dicts[pname]["node_type_id"]
    n_edges = len(source_ids) * len(target_ids)
    count = 0
    type_dict = {
        "edge_type_id": [],
        "edge_group_id": [0] * n_edges,
        "edge_group_index": [0] * n_edges,
        "syn_weight": [],
    }
    for i in source_ids:
        for j in target_ids:
            type_dict["edge_type_id"].append(count)
            type_dict["syn_weight"].append(weight_dicts[pname][j + i * 4])
            count += 1
    df = pd.DataFrame(type_dict)
    df.to_csv(f"network/{pname}_l23_edge_types.csv", index=False, sep=" ")
    create_edge_h5(pname, "l23", type_dict["edge_type_id"])

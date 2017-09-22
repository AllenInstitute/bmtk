import h5py
from .. import Network

def write_edges_to_h5(network, filename, synapse_key=None, verbose=True):
    assert(isinstance(network, Network))

    # The network edges may either be a raw value, dictionary or list
    if synapse_key == None:
        lookup = lambda x: x

    elif isinstance(synapse_key, str):
        lookup = lambda x: x[synapse_key]

    elif isinstance(synapse_key, int):
        lookup = lambda x: x[synapse_key]

    else:
        raise Exception("Unable to resolve the synapse_key type.")

    # Create the tables for indptr, nsyns and src_gids
    if verbose:
        print("> building tables with {} nodes and {} edges.".format(network.nnodes, network.nedges))
    indptr_table = [0]
    nsyns_table = []
    src_gids_table = []
    for trg in network.nodes():
        # TODO: check the order of the node list
        tid = trg[1]['id']
        for edges in network.edges([tid], rank=1):
            src_gids_table.append(edges[0])
            nsyns_table.append(lookup(edges[2]))

        if len(src_gids_table) == indptr_table[-1]:
            print "node %d doesn't have any edges" % (tid)
        indptr_table.append(len(src_gids_table))

    # Save the tables in h5 format
    if verbose:
        print("> Saving table to {}.".format(filename))
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('indptr', data=indptr_table)
        hf.create_dataset('nsyns', data=nsyns_table)
        hf.create_dataset('src_gids', data=src_gids_table, dtype=int32)
        hf.attrs["shape"] = (network.nnodes, network.nnodes)

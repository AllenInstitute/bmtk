# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import h5py
from ..network import Network

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
            print("node %d doesn't have any edges {}".format(tid))
        indptr_table.append(len(src_gids_table))

    # Save the tables in h5 format
    if verbose:
        print("> Saving table to {}.".format(filename))
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('indptr', data=indptr_table)
        hf.create_dataset('nsyns', data=nsyns_table)
        hf.create_dataset('src_gids', data=src_gids_table, dtype=int32)
        hf.attrs["shape"] = (network.nnodes, network.nnodes)

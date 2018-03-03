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
import itertools
import functools
import types


class IteratorCache(object):
    def __init__(self):
        self.cache = {}

    def create(self, itr_name, itr_type, **params):
        if params is None:
            params = {}

        if (itr_name, itr_type) in self.cache:
            func = self.cache[(itr_name, itr_type)]
            return functools.partial(func, **params)

        else:
            raise Exception("Couldn't find iterator for ({}, {}).".format(itr_name, itr_type))

    def register(self, name, itr_type, func):
        self.cache[(name, itr_type)] = func


def create(iterator, connector, **params):
    return ITERATOR_CACHE.create(iterator, type(connector), **params)


def register(name, dtype, func):
    ITERATOR_CACHE.register(name, dtype, func)


########################################################################
# Pre-defined iterators
########################################################################
def one_to_all_iterator(source_nodes, target_nodes, connector):
    """Calls the connector function with (1 source, all targets), iterated for each source"""
    target_list = list(target_nodes)  # list of all targets
    target_node_ids = [t.node_id for t in target_list]  # slight improvement than calling node_id S*T times
    for source in source_nodes:
        source_node_id = source.node_id
        edge_vals = connector(source, target_list)
        for i, target in enumerate(target_list):
            yield (source_node_id, target_node_ids[i], edge_vals[i])


def all_to_one_iterator(source_nodes, target_nodes, connector):
    """Iterate through all the target nodes and return target node + list of all sources"""
    source_list = list(source_nodes)
    for target in target_nodes:
        val = connector(source_list, target)
        for i, source in enumerate(source_list):
            yield (source.node_id, target.node_id, val[i])


def one_to_one_iterator(source_nodes, target_nodes, connector):
    # TODO: may be faster to pull out the node_ids, don't user itertools
    for source, target in itertools.product(source_nodes, target_nodes):
        val = connector(source, target)
        yield (source.node_id, target.node_id, val)


def one_to_one_list_iterator(source_nodes, target_nodes, vals):
    assert(len(vals) == len(source_nodes)*len(target_nodes))
    for i, (source, target) in enumerate(itertools.product(source_nodes, target_nodes)):
        yield (source.node_id, target.node_id, vals[i])


def one_to_all_list_iterator(source_nodes, target_nodes, vals):
    assert(len(vals) == len(target_nodes))
    source_ids = [s.node_id for s in list(source_nodes)]
    target_ids = [t.node_id for t in list(target_nodes)]
    for src_id in source_ids:
        for i, trg_id in enumerate(target_ids):
            yield (src_id, trg_id, vals[i])


def all_to_one_list_iterator(source_nodes, target_nodes, vals):
    assert(len(vals) == len(source_nodes))
    source_ids = [s.node_id for s in list(source_nodes)]
    target_ids = [t.node_id for t in list(target_nodes)]
    for trg_id in target_ids:
        for i, src_id in enumerate(source_ids):
            yield (src_id, trg_id, vals[i])


def lambda_iterator(source_nodes, target_nodes, lambda_val):
    for source, target in itertools.product(source_nodes, target_nodes):
        yield (source.node_id, target.node_id, lambda_val())


ITERATOR_CACHE = IteratorCache()
register('one_to_one', functools.partial, one_to_one_iterator)
register('all_to_one', functools.partial, all_to_one_iterator)
register('one_to_all', functools.partial, one_to_all_iterator)

register('one_to_one', list, one_to_one_list_iterator)
register('one_to_all', list, one_to_all_list_iterator)
register('all_to_one', list, all_to_one_list_iterator)


register('one_to_one', types.FunctionType, lambda_iterator)

import math


def gaussianLL(trg_node, src_node, edge_prop):
    src_tuning = src_node['tuning_angle']
    tar_tuning = trg_node['tuning_angle']

    w0 = edge_prop["weight_max"]
    sigma = edge_prop["weight_sigma"]
  
    delta_tuning = abs(abs(abs(180.0 - abs(float(tar_tuning) - float(src_tuning)) % 360.0) - 90.0) - 90.0)
    weight = w0*math.exp(-(delta_tuning / sigma) ** 2)

    return weight * edge_prop['nsyns']


def wmax(trg_node, src_node, edge_prop):
    return edge_prop["weight_max"] * edge_prop['nsyns']

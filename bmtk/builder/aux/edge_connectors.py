import numpy as np
import random


def distance_connector(source, target, d_weight_min, d_weight_max, d_max, nsyn_min, nsyn_max):
    # Avoid self-connections.
    sid = source.node_id
    tid = target.node_id
    if sid == tid:
        return None

    # first create weights by euclidean distance between cells
    r = np.linalg.norm(np.array(source['positions']) - np.array(target['positions']))
    if r > d_max:
        dw = 0.0
    else:
        t = r / d_max
        dw = d_weight_max * (1.0 - t) + d_weight_min * t

    # drop the connection if the weight is too low
    if dw <= 0:
        return None

    # filter out nodes by treating the weight as a probability of connection
    if random.random() > dw:
        return None

    # Add the number of synapses for every connection.
    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    return tmp_nsyn


def connect_random(source, target, nsyn_min=0, nsyn_max=10, distribution=None):
    return np.random.randint(nsyn_min, nsyn_max)
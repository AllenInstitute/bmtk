import math


def gaussianLL(tar_prop,src_prop,con_prop):
    src_tuning = src_prop['tuning_angle']
    tar_tuning = tar_prop['tuning_angle']

    w0 = con_prop["weight_max"]
    sigma = con_prop["weight_sigma"]
  
    delta_tuning = abs(abs(abs(180.0 - abs(float(tar_tuning) - float(src_tuning)) % 360.0) - 90.0) - 90.0)
    weight = w0*math.exp(-(delta_tuning / sigma) ** 2)

    return weight


def wmax(tar_prop,src_prop,con_prop):
    w0 = con_prop["weight_max"]
    return w0

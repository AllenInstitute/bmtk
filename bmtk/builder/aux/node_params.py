import numpy as np
import math


def positions_columinar(N=1, center=[0.0, 50.0, 0.0], height=100.0, min_radius=0.0, max_radius=1.0, distribution='uniform'):
    phi = 2.0 * math.pi * np.random.random([N])
    r = np.sqrt((min_radius**2 - max_radius**2) * np.random.random([N]) + max_radius**2)
    x = center[0] + r * np.cos(phi)
    z = center[2] + r * np.sin(phi)
    y = center[1] + height * (np.random.random([N]) - 0.5)

    return np.column_stack((x, y, z))


def xiter_random(N=1, min_x=0.0, max_x=1.0):
    return np.random.uniform(low=min_x, high=max_x, size=(N,))
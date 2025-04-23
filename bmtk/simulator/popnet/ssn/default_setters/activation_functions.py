from bmtk.simulator.popnet.ssn.pyfunction_cache import add_activation_function

try:
    from numba import njit

except ImportError as ie:
    from bmtk.simulator.popnet.ssn.utils import empty_decorator
    njit = empty_decorator


@njit
def relu2(array):
    # if the element is negative, set it to zero using loop
    # destructive method (alters the original array), but faster than the above one.
    for i in range(len(array)):
        if array[i] < 0:
            array[i] = 0
    return array


add_activation_function(relu2, name='default', overwrite=False)
add_activation_function(relu2, overwrite=False)

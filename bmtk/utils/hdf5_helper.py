import sys

if sys.version_info[0] == 3:
    using_py3 = True
    range_itr = range
else:
    using_py3 = False
    range_itr = xrange


def get_attribute_h5(h5obj, attribut_name, default=None):
    val = h5obj.attrs.get(attribut_name, default)
    if using_py3 and isinstance(val, bytes):
        # There is an but with h5py returning unicode/str based attributes as bytes
        val = val.decode()

    return val

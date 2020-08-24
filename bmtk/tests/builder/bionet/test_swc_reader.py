import pytest
import tempfile
import numpy as np

nrn = pytest.importorskip('neuron')  # skip tests if neuron isn't installed on env.

from bmtk.builder.bionet.swc_reader import SWCReader


nr5a1_morphology = """##n,type,x,y,z,radius,parent
1 1 -0.0000 0.0000 -0.0000 6.4406 -1
2 3 -3.5283 -4.3649 0.3294 0.3686 1
3 3 -3.8706 -5.0094 2.0525 0.5339 2
4 3 -4.1573 -6.0541 2.5315 0.661 3
5 3 -4.9345 -6.9190 2.2990 0.6991 4
918 4 -0.5971 5.3276 0.4139 0.1907 1
919 4 -0.8745 6.4601 0.5746 0.2669 918
920 4 -1.3117 7.5517 0.9188 0.3178 919
921 4 -1.7148 8.6488 1.1361 0.3432 920
1493 2 -2.2165 -4.7239 0.8437 0.1907 1
1494 2 -2.7098 -5.7699 0.7002 0.2669 1493
1495 2 -3.1899 -6.8480 0.1199 0.3305 1494
1496 2 -3.5848 -7.9629 -0.5080 0.3813 1495
1497 2 -3.9590 -9.0850 -1.1323 0.4068 1496
"""


def test_swc_reader():
    tmpfile = tempfile.NamedTemporaryFile(suffix='.h5')
    with open(tmpfile.name, 'w') as f:
        f.write(nr5a1_morphology)

    swc_reader = SWCReader(tmpfile.name)
    sec_ids, sec_xs = swc_reader.choose_sections(['soma'], [0, 10.0])  # randomly choose sec_ids
    assert(np.all(sec_ids == [0]))
    assert(np.all(sec_xs == [0.5]))
    assert(np.allclose(swc_reader.get_dist(sec_ids), [6.4406], 0.001))
    assert(np.allclose(swc_reader.get_coord(sec_ids, sec_xs, soma_center=(0.0, 0.0, 0.0)), [0.0, 0.0, 0.0]))
    assert(np.all(swc_reader.get_type(sec_ids) == [1]))

    sec_ids, sec_xs = swc_reader.choose_sections(['dend'], [0, 100.0])  # randomly choose sec_ids
    # print(sec_ids, sec_xs)
    assert(np.all(sec_ids == [1]))
    assert(np.all(sec_xs == [0.5]))
    assert(np.allclose(swc_reader.get_dist(sec_ids), [8.5613], 0.001))
    assert(np.allclose(swc_reader.get_coord(sec_ids, sec_xs, soma_center=(0.0, 0.0, 0.0)),
                       [-3.87059999, -5.00939989, -3.87059999]))
    assert(swc_reader.get_type(sec_ids) == [3])


if __name__ == '__main__':
    test_swc_reader()

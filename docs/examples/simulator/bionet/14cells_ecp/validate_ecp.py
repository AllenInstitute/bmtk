import h5py
import numpy as np
from numpy.random import randint

SAMPLE_SIZE=100

expected_h5 = h5py.File('expected/ecp.h5', 'r')
nrows, ncols = expected_h5['ecp'].shape
expected_mat = np.matrix(expected_h5['ecp'])

results_h5 = h5py.File('output/ecp.h5', 'r')
assert('ecp' in results_h5.keys())
results_mat = np.matrix(results_h5['ecp'][:])

assert(results_h5['ecp'].shape == (nrows, ncols))
for i, j in zip(randint(0, nrows, size=SAMPLE_SIZE), randint(0, ncols, size=SAMPLE_SIZE)):
    assert(results_mat[i, j] == expected_mat[i, j])




#print np.random.choice(range(0, 86), size=10)

#print
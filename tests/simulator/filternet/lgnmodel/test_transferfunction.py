import pytest
import numpy as np
import sympy.abc

from bmtk.simulator.filternet.lgnmodel import transferfunction


def test_heaviside():
    tf = transferfunction.ScalarTransferFunction('Heaviside(s+1.05)*(s+1.05)', symbol=sympy.abc.s)

    assert(tf(1.0) == 1.0+1.05)
    assert(tf(0.0) == 1.05)
    assert(tf(-1.049) > 0)
    assert(tf(-1.05) == 0)
    assert(tf(-2.0) == 0)


if __name__ == '__main__':
    # test_heaviside()
    tf = transferfunction.ScalarTransferFunction('Heaviside(s+1.05)*(s+1.05)', symbol=sympy.abc.s)
    tf.imshow(np.linspace(-10.0, 10.0), np.linspace(-10.0, 10.0))

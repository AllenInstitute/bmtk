from sympy.utilities.lambdify import lambdify
from sympy import Matrix
import sympy.parsing.sympy_parser as symp
import sympy.abc
import numpy as np


class ScalarTransferFunction(object):
    def __init__(self, transfer_function_string, symbol=sympy.abc.s):
        self.symbol = symbol
        self.transfer_function_string = transfer_function_string
        # replacing sympy.Heaviside() with np.heaviside() for better performance
        modules = [{'Heaviside': lambda x, y=0.5: np.heaviside(x, 0.5)}, 'numpy', 'sympy']
        self.closure = lambdify(self.symbol, symp.parse_expr(self.transfer_function_string), modules=modules)

    def __call__(self, s):
        return self.closure(s)
    
    def to_dict(self):
        return {'class': (__name__, self.__class__.__name__), 'function': self.transfer_function_string}
        
    def imshow(self, rates, times=None, show=True):
        import matplotlib.pyplot as plt

        vals = [self(rate) for rate in rates]
        times = np.linspace(0.0, 1.0, len(rates)) if times is None else times

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time (seconds)')
        ax1.set_ylabel(str(self.symbol), color='b')
        ax1.plot(times, rates, '--b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.set_ylabel('transform', color='r')
        ax2.plot(times, vals, 'r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(self.transfer_function_string)
        if show:
            plt.show()
        

class MultiTransferFunction(object):
    def __init__(self, symbol_tuple, transfer_function_string):
        self.symbol_tuple = symbol_tuple
        self.transfer_function_string = transfer_function_string
        modules = [{'Heaviside': lambda x, y=0.5: np.heaviside(x, 0.5)}, 'numpy', 'sympy']
        self.closure = lambdify(self.symbol_tuple,symp.parse_expr(self.transfer_function_string), modules=modules)

    def __call__(self, *s):
        if isinstance(s[0], (float,)):
            return self.closure(*s)
        else:
            return np.array(list(map(lambda x: self.closure(*x), zip(*s))))
    
    def to_dict(self):
        return {'class': (__name__, self.__class__.__name__), 'function': self.transfer_function_string}

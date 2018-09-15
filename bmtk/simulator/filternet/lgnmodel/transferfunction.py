from sympy.utilities.lambdify import lambdify
import sympy.parsing.sympy_parser as symp
import sympy.abc
import numpy as np


class ScalarTransferFunction(object):
    def __init__(self, transfer_function_string, symbol=sympy.abc.s):
        self.symbol = symbol
        self.transfer_function_string = transfer_function_string
        self.closure = lambdify(self.symbol, symp.parse_expr(self.transfer_function_string), modules=['sympy'])
        
    def __call__(self, s):
        return self.closure(s)
    
    def to_dict(self):
        return {'class': (__name__, self.__class__.__name__),
                'function': self.transfer_function_string}
        
    def imshow(self, xlim, ax=None, show=True, save_file_name=None, ylim=None):
        # TODO: This function should be removed (as Ram to see if/where it's used) since it will fail (no t_vals)
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        plt.plot(self.t_vals, self.kernel)
        ax.set_xlabel('Time (Seconds)')
        
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xlim is not None:
            ax.set_xlim((self.t_range[0], self.t_range[-1]))
        
        if save_file_name is not None:
            plt.savefig(save_file_name, transparent=True)
        
        if show:
            plt.show()
        
        return ax


class MultiTransferFunction(object):
    def __init__(self, symbol_tuple, transfer_function_string):
        self.symbol_tuple = symbol_tuple
        self.transfer_function_string = transfer_function_string
        self.closure = lambdify(self.symbol_tuple, symp.parse_expr(self.transfer_function_string), modules=['sympy'])

    def __call__(self, *s):
        if isinstance(s[0], (float,)):
            return self.closure(*s)
        else:
            return np.array(list(map(lambda x: self.closure(*x), zip(*s))))
    
    def to_dict(self):
        return {'class': (__name__, self.__class__.__name__),
                'function': self.transfer_function_string}

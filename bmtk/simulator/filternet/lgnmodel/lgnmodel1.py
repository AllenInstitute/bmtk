import numpy as np
import matplotlib.pyplot as plt

def line_plot(evaluate_result, ax=None, show=True, save_file_name=None, xlabel=None, plotstyle=None):

        if ax is None:
            _, ax = plt.subplots(1,1)

        if not plotstyle is None:
            for ((t_range, y_vals), curr_plotstyle) in zip(evaluate_result, plotstyle):   
                ax.plot(t_range, y_vals, curr_plotstyle)
        else:
            for t_range, y_vals in evaluate_result:   
                ax.plot(t_range, y_vals)

        if xlabel is None:
            ax.set_xlabel('Time (Seconds)')
        else:
            ax.set_xlabel(xlabel)
            
        if xlabel is None:
            ax.set_xlabel('Firing Rate (Hz)')
        else:
            ax.set_xlabel(xlabel)
         
        if not save_file_name is None:
            plt.savefig(save_file_name, transparent=True)
            

            
             
        if show == True:
            plt.show() 

def heat_plot(evaluate_result, ax=None, show=True, save_file_name=None, colorbar=True, **kwargs):
    
        if ax is None:
            _, ax = plt.subplots(1,1)

        data = np.empty((len(evaluate_result), len(evaluate_result[0][0])))
        for ii, (t_vals, y_vals) in enumerate(evaluate_result):   
            data[ii,:] = y_vals
            
        cax = ax.pcolor(t_vals, np.arange(len(evaluate_result)), data, **kwargs)
        ax.set_ylim([0,len(evaluate_result)-1])
        ax.set_xlim([t_vals[0], t_vals[-1]])
        ax.set_ylabel('Neuron id')
        ax.set_xlabel('Time (Seconds)')
        
        if colorbar == True:
            plt.colorbar(cax)
         
        if not save_file_name is None:
            plt.savefig(save_file_name, transparent=True)
             
        if show == True:
            plt.show() 
        
        
            

class LGNModel(object):
    def __init__(self, cell_list):
        self.cell_list = cell_list
        
    def evaluate(self, movie, **kwargs):
        return [cell.evaluate(movie, **kwargs) for cell in self.cell_list]

    def __len__(self):
        return len(self.cell_list)

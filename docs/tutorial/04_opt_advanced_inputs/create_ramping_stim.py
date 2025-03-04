import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


time = np.linspace(0.0, 3000.0, 1000)
# amplitude = time*0.0003-.100
amplitude = time*0.0002-.100


pd.DataFrame({
    'time': time,
    'amplitude': amplitude
}).to_csv('inputs/ramping_xstim.csv', sep=' ', index=False)
plt.plot(time, amplitude)
plt.show()
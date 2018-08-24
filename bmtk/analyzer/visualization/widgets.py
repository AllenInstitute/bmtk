# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp
import numpy as np

class PlotWidget(object):
    
    def __init__(self, t_range, y_range, rate_ax=None, position_ax=None, metadata={}, location_markersize=5):
        
        if rate_ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = rate_ax
        self.position_ax = position_ax 
        
        self.t_range = t_range
        self.y_range = y_range
        self.interp_fcn = spinterp.interp1d(self.t_range, self.y_range)
        self._t = None
        self.metadata=metadata
        self.artist_list = []
        self.location_markersize = location_markersize
        
    @property
    def y(self):
        return self.interp_fcn(self._t)
        
    def initialize(self, t0, **kwargs):
        
        self._t = t0
        self.plot_data, = self.ax.plot(self.t_range,self.y_range,**kwargs)
        self.vertical_rule_data, = self.ax.plot([self._t, self._t],self.ax.get_ylim(),'--r')
        self.point_data, = self.ax.plot([self._t],[self.y],'*r')
        
        self.artist_list = [self.plot_data, self.vertical_rule_data, self.point_data]
        
        if (not self.position_ax is None) and 'position' in self.metadata:
            x = self.metadata['position'][0]
            y = self.metadata['position'][1]
            self.location_point_data, = self.position_ax.plot([x],[y],'*r', markersize=self.location_markersize)
            self.artist_list.append(self.location_point_data)
            

    def update(self, t):
        
        self._t = t
        self.point_data
        self.point_data.set_xdata([self._t])
        self.vertical_rule_data.set_xdata([self._t, self._t])
        self.vertical_rule_data.set_ydata(self.ax.get_ylim())
        
        for data in self.artist_list:
            self.ax.figure.canvas.blit(data)
        
    def set_visible(self, visible_or_not):


        for data in self.artist_list:
            data.set_visible(visible_or_not)
            self.ax.figure.canvas.blit(data)
        

class MovieWidget(object):
    
    def __init__(self, t_range, data, ax=None, metadata={}):
        
        if ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax
        
        self.t_range = t_range
        self.frame_rate = 1./np.mean(np.diff(t_range))
        self.data = data
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.metadata=metadata
        
    def initialize(self, t0, vmin=-1, vmax=1, cmap=plt.cm.gray):
        
        data = self.data[self.ti(t0),:,:]
        self.im = self.ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)

    def update(self, t):
        
        data = self.data[self.ti(t),:,:]
        self.im.set_data(data)
        self.ax.figure.canvas.draw()
        
    def ti(self, t):
        return int(t*self.frame_rate) - int(self.t_range[0]*self.frame_rate)
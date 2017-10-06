# -*- coding: utf-8 -*-

"""Builds and simulates a 14 cell V1 example using AI/BBP network format."""

import os, sys

import bmtk.simulator.bionet.config as config
#from bmtk.simulator.bionet.biograph import BioGraph
#from bmtk.simulator.bionet.property_schemas import AIPropertySchema
import bmtk.analyzer.spikes_loader as spklod
import bmtk.analyzer.spikes_analyzer as spkan



import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import numpy as np


config_file = "config.json"     # Get config from the command line argument
conf = config.from_json(config_file)

spikes_file_name = conf["output"]["spikes_h5"]
#spikes_file_name = conf["output"]["spikes_ascii"]

t1=time.time()
spikes_v1 = spklod.load_spikes(spikes_file_name)
t2=time.time()
print "load time:",t2-t1

tstop = conf["run"]["tstop"]

ncells=14
node_ids = np.arange(ncells)


mean_firing_rates = spkan.get_mean_firing_rates(spikes_v1["gid"], node_ids, tstop)

fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[3, 1]},sharey=True)

ax[0].scatter(spikes_v1["time"],spikes_v1["gid"], marker=".", lw=0, s=10,facecolors="black") 
ax[0].axis('tight')

ax[0].set_ylabel('node_id');
ax[0].set_xlabel('time (ms)');
ax[0].set_xlabel('time (ms)');

ax[0].set_ylim([np.min(node_ids), np.max(node_ids)])
ax[0].set_title('V1 raster');

ax[1].scatter(mean_firing_rates,node_ids, marker=".", lw=0, s=20,facecolors="black")
ax[1].set_xlabel('rate (Hz)');
ax[1].set_title('mean_firing_rate');

fig2, ax2 = plt.subplots(2,1)

t1=time.time()
spikes_file_name = conf["input"][0]["file"]
print spikes_file_name
spikes_lgn = spklod.load_spikes(spikes_file_name,"trial_0")
t2=time.time()
print "load time:",t2-t1

ax2[0].scatter(spikes_lgn["time"],spikes_lgn["gid"], marker=".", lw=0, s=10,facecolors="blue") 
ax2[0].set_ylabel("node_id")
ax2[0].set_title('LGN raster');
ax2[0].axis('tight')

t1=time.time()
spikes_file_name = conf["input"][1]["file"]
print spikes_file_name
spikes_tw = spklod.load_spikes(spikes_file_name,"trial_0")
t2=time.time()
print "load time:",t2-t1

ax2[1].scatter(spikes_tw["time"],spikes_tw["gid"], marker=".", lw=0, s=10,facecolors="red") 
ax2[1].set_ylabel("node_id")
ax2[1].set_xlabel('time (ms)');
ax2[1].set_title('TW raster');
ax2[1].axis('tight')


plt.show()


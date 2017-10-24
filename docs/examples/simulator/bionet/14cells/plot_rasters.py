# plot spikes

import os, sys

import bmtk.simulator.bionet.config as config
import bmtk.analyzer.spikes_loader as spkload
import bmtk.analyzer.spikes_analyzer as spkan
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time
import numpy as np


config_file = "config.json"     # Get config from the command line argument
conf = config.from_json(config_file)

spikes_file_name = conf["output"]["spikes_hdf5_file"]
#spikes_file_name = conf["output"]["spikes_ascii"]

t1=time.time()
spikes_v1 = spkload.load_spikes(spikes_file_name)
t2=time.time()
print "load time:",t2-t1

tstop = conf["run"]["tstop"]

ncells=14
node_ids = np.arange(ncells)


fig, ax = plt.subplots(1,1)

ax.scatter(spikes_v1["time"],spikes_v1["gid"], marker=".", lw=0, s=20,facecolors="blue") 
ax.axis('tight')

ax.set_ylabel('node_id');
ax.set_xlabel('time (ms)');
ax.set_xlabel('time (ms)');

ax.set_ylim([np.min(node_ids), np.max(node_ids)])
ax.set_title('V1 cells');
ax.set_ylim([-0.5,13.5]);

fig2, ax2 = plt.subplots(2,1)

t1=time.time()
spikes_file_name = conf["input"][0]["file"]
print spikes_file_name
spikes_lgn = spkload.load_spikes(spikes_file_name,"trial_0")
t2=time.time()
print "load time:",t2-t1

ax2[0].scatter(spikes_lgn["time"],spikes_lgn["gid"], marker=".", lw=0, s=10,facecolors="green") 
ax2[0].set_ylabel("node_id")
ax2[0].set_title('LGN cells');
ax2[0].axis('tight')

t1=time.time()
spikes_file_name = conf["input"][1]["file"]
print spikes_file_name
spikes_tw = spkload.load_spikes(spikes_file_name,"trial_0")
t2=time.time()
print "load time:",t2-t1

ax2[1].scatter(spikes_tw["time"],spikes_tw["gid"], marker=".", lw=0, s=10,facecolors="firebrick") 
ax2[1].set_ylabel("node_id")
ax2[1].set_xlabel('time (ms)');
ax2[1].set_title('TW cells');
ax2[1].axis('tight')


plt.show()


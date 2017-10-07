# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
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
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import pandas as pd
import time
import datetime
import os
import shutil
import json
import numpy as np
import logging
import h5py

from neuron import h

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import nrn


pc = h.ParallelContext()    # object to access MPI methods
MPI_Rank = int(pc.id())


def load_json(fullpath):
    """Tries to load a json file

    :param fullpath: path to file name to be loaded
    :return: json dictionary object
    """
    try:    
        with open(fullpath, 'r') as f:
            data = json.load(f)
            return data
    except IOError:
        print2log0("ERROR: cannot open {}".format(fullpath))
        nrn.quit_execution()


def load_h5(file_name):
    """load hdf5 file into memory

    :param file_name: full path to h5 file
    :return: file handle to hdf5 object
    """
    assert(os.path.exists(file_name))
    try:    
        data_handle = h5py.File(file_name, 'r')
        return data_handle
    except IOError:
        print2log0("ERROR: cannot open {}".format(file_name))
        nrn.quit_execution()


def load_csv(fullpath):
    """Load a csv file

    :param fullpath: path to the file name to be loaded
    :return: pandas dataframe
    """
    try:
        data = pd.read_csv(fullpath, sep=' ')
        return data
    except IOError:
        print2log0("ERROR: cannot open {}".format(fullpath))
        nrn.quit_execution()


def create_log(conf):
    logging.basicConfig(filename=conf["output"]["log_file"], level=logging.DEBUG)
    print2log0now('Created a log file')


def print2log(message):
    """Print statements to the log file from all processors"""
    delta_t = time.clock()
    full_string = '{}, t_wall: {} s'.format(message, str(delta_t))
    logging.info(full_string) 


def print2log0(string): 
    """Print from rank=0 only"""
    # CAUTION: NEURON's warning will not be save in either RUN_LOG or the log file created by the PBS.
    # To catch possible NEURON warnings, you need to use shell redirection, e.g.: $ python foo.py > file
    if int(pc.id()) == 0:
        delta_t = time.clock()
        # delta_t = timeit.default_timer()
        full_string = string + ' -- t_wall: %s s' % (str(delta_t))
        print(full_string)   # echo on the screen
        logging.info(full_string) 


def print2log0now(string):
    """Print from rank=0 only and report date,time"""
    if int(pc.id()) == 0:
        now = datetime.datetime.now()

        full_string = string + ' -- on %02d/%02d/%02d at %02d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(full_string)   # echo on the screen
        logging.info(full_string) 


def extend_output_files(gids):
    # TODO: resize the files when running from an existing state
    pass


def create_output_files(conf, gids):
    if conf["run"]["calc_ecp"]:  # creat single file for ecp from all contributing cells
        print2log0('    Will save time series of the ECP!')
        create_ecp_file(conf)

    if conf["run"]["save_cell_vars"]:
        print2log0('    Will save time series of individual cells')
        create_cell_vars_files(conf, gids)
                
    create_spike_file(conf, gids)  # a single file including all gids
    

def create_ecp_file(conf):
    """a single ecp file for the entire network"""

    dt = conf["run"]["dt"]
    tstop = conf["run"]["tstop"]
    nsteps = int(round(tstop/dt))
    nsites = conf['run']['nsites']

    ofname = conf["output"]["ecp_file"]
    if int(pc.id()) == 0:  # create single file for ecp from all contributing cells
        with h5py.File(ofname, 'w') as f5:
            f5.create_dataset('ecp', (nsteps, nsites), maxshape=(None, nsites), chunks=True)
            f5.attrs['dt'] = dt
            f5.attrs['tstart'] = 0.0
            f5.attrs['tstop'] = tstop

    pc.barrier()


def create_cell_vars_files(conf, gids):
    """create 1 hfd5 files per gid"""

    dt = conf["run"]["dt"]
    tstop = conf["run"]["tstop"]

    nsteps = int(round(tstop/dt))

    for gid in gids["save_cell_vars"]:
        ofname = conf["output"]["cell_vars_dir"]+'/%d.h5' % (gid)
        with h5py.File(ofname, 'w') as h5:
            h5.attrs['dt'] = dt
            h5.attrs['tstart'] = 0.0
            h5.attrs['tstop'] = tstop

            for var in conf["run"]["save_cell_vars"]:
                h5.create_dataset(var, (nsteps,), maxshape=(None,), chunks=True)

            h5.create_dataset('spikes', (0,), maxshape=(None,), chunks=True)

            if conf["run"]["calc_ecp"]:  # then also create a dataset for the ecp
                nsites = conf['run']['nsites']
                h5.create_dataset('ecp', (nsteps, nsites), maxshape=(None, nsites), chunks=True)


def create_spike_file(conf, gids_on_rank):
    """create a single hfd5 files for all gids"""
    print2log0('    Will save spikes')

    ofname = conf["output"]["spikes_hdf5_file"]
    tstop = conf["run"]["tstop"]

    if int(pc.id()) == 0:  # create h5 file
        with h5py.File(ofname, 'w') as h5:
            h5.attrs['tstart'] = 0.0
            h5.attrs['tstop'] = tstop
            h5.create_dataset("time", shape=(0,), maxshape=(None,), chunks=True)
            h5.create_dataset("gid", shape=(0,), maxshape=(None,), chunks=True, dtype=np.int32)

    pc.barrier()

    if int(pc.id()) == 0:  # create ascii file
        ofname = conf["output"]["spikes_ascii_file"]
        f = open(ofname, 'w')  # create ascii file
        f.close()

    pc.barrier()


def get_spike_trains_handle(file_name, trial_name):
    f5 = load_h5(file_name)
    spike_trains_handle = f5['processing/%s/spike_train' % trial_name]
    return spike_trains_handle


def setup_output_dir(conf):

    start_from_state =False
    if start_from_state:  # starting from a previously saved state
        try:
            assert os.path.exists(conf["output"]["output_dir"])
            print2log0('Will run simulation from a previously saved state...')
        except:
            print('ERROR: directory with the initial state does not exist')
            nrn.quit_execution()

    elif not start_from_state:  # starting from a new (init) state
        if int(pc.id()) == 0:
            if os.path.exists(conf["output"]["output_dir"]):
                if conf["run"]['overwrite_output_dir']:
                    shutil.rmtree(conf["output"]["output_dir"])
                    print('Overwriting the output directory %s:' %conf["output"]["output_dir"]) # must print to stdout because the log file is not created yet
                else:
                    print('ERROR: Directory already exists')
                    print("To overwrite existing output_dir set 'overwrite_output_dir': True")
                    nrn.quit_execution()

            os.makedirs(conf["output"]["output_dir"])
            os.makedirs(conf["output"]["cell_vars_dir"])
#            os.makedirs(conf["output"]["state_dir"])

            create_log(conf)
            config.copy(conf)

        pc.barrier()

    print2log0('Output directory: %s' % conf["output"]["output_dir"])
    print2log0('Config file: %s' % conf["config_path"])


def save_block_to_disk(conf, data_block, time_step_interval):
    """save data in blocks to hdf5"""
    save_ecp(conf, data_block, time_step_interval)
    save_cell_vars(conf, data_block, time_step_interval)
    save_spikes2h5(conf, data_block)
    save_spikes2ascii(conf, data_block)


def save_spikes2h5(conf, data_block):
    """Save spikes to h5 file: into time,gid datasets

    Spike times are not necessarily in order. For comparison between runs, need to sort spikes

    :param conf:
    :param data_block:
    """
    spikes = data_block["spikes"]
    ofname = conf["output"]["spikes_hdf5_file"]
    ranks = xrange(int(pc.nhost()))
    for rank in ranks:  # iterate over the ranks
        if rank == int(pc.id()):  # wait until finished with a particular rank
            with h5py.File(ofname, 'a') as h5:
                for gid in spikes:                          # save spikes of all gids on this rank
                    nspikes_saved = h5["gid"].shape[0]      # find number of spikes already in the file
                    nspikes_to_add = len(spikes[gid])          # find number of spikes to add
                    nspikes = nspikes_saved+nspikes_to_add     # total number of spikes
                    h5["time"].resize((nspikes,))         # resize the dataset
                    h5["gid"].resize((nspikes,))         # resize the dataset
                    h5["time"][nspikes_saved:nspikes] = np.array(spikes[gid])
                    h5["gid"][nspikes_saved:nspikes] = np.array([gid]*nspikes_to_add)

        pc.barrier()    # move on to next rank


def save_ecp(conf, data_block, time_step_interval):
    """Save ECP from each rank to disk into a single file"""
    itstart, itend = time_step_interval
    ofname = conf["output"]["ecp_file"]
    if conf["run"]['calc_ecp']:

        ranks = xrange(int(pc.nhost()))
        for rank in ranks:              # iterate over the ranks
            if rank == int(pc.id()):      # wait until finished with a particular rank
                with h5py.File(ofname, 'a') as f5:
                    f5["ecp"][itstart:itend, :] += data_block['ecp'][0:itend-itstart, :]
                    f5.attrs["tsave"] = data_block["tsave"]  # update tsave
                    data_block['ecp'][:] = 0.0

            pc.barrier()    # move on to next rank


def save_cell_vars(conf, data_block, time_step_interval):
    """save to disk with one file per gid"""
    itstart, itend = time_step_interval
    for gid, cell_data_block in data_block['cells'].items():
        ofname = conf["output"]["cell_vars_dir"]+'/%d.h5' % (gid)

        with h5py.File(ofname, 'a') as h5:

            h5.attrs["tsave"] = data_block["tsave"]  # update tsave
            for var in conf["run"]["save_cell_vars"]:
                h5[var][itstart:itend] = cell_data_block[var][0:itend-itstart]
                cell_data_block[var][:] = 0.0

            spikes = data_block["spikes"]
            nspikes_saved = h5["spikes"].shape[0]   # find number of spikes
            nspikes_add = len(spikes[gid])     # find number of spikes to add
            nspikes = nspikes_saved+nspikes_add     # total number of spikes
            h5["spikes"].resize((nspikes,))         # resize the dataset
            h5["spikes"][nspikes_saved:nspikes] = np.array(spikes[gid])  # save hocVector as a numpy arrray

            if "ecp" in cell_data_block.keys():
                h5["ecp"][itstart:itend, :] = cell_data_block['ecp'][0:itend-itstart, :]
                cell_data_block['ecp'][:] = 0.0


def save_spikes2ascii(conf, data_block):
    """Save spikes to ascii file as tuples (t,gid)"""
    spikes = data_block["spikes"]
    ofname = conf["output"]["spikes_ascii_file"]

    ranks = xrange(int(pc.nhost()))
    for rank in ranks:
        if rank == int(pc.id()):
            f = open(ofname, 'a')
            for gid in spikes:
                tVec = spikes[gid]
                for t in tVec:
                    f.write('%.3f %d\n' % (t, gid))
            f.close()
        pc.barrier()


def save_state(conf):
    state = h.SaveState()
    state.save()

    state_dir = conf["output"]["state_dir"]
    f = h.File('{}/state_rank-{}'.format(state_dir, int(pc.id())))
    # f = h.File(state_dir + '/state_rank-%d' % (int(pc.id())))
    state.fwrite(f, 0)
    rlist = h.List('Random')
    for r_tmp in rlist:
        f.printf('%g\n', r_tmp.seq())
    f.close()


def read_state(conf):
    state_dir = conf["output"]["state_dir"]

    state = h.SaveState()
    f = h.File('{}/state_rank-{}'.format(state_dir, int(pc.id())))
    # f = h.File(state_dir+'/state_rank-%d' % (int(pc.id())))
    state.fread(f, 0)
    state.restore()
    rlist = h.List('Random')
    for r_tmp in rlist:
        r_tmp.seq(f.scanvar())
    f.close()

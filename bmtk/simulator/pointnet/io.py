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
"""
Functions for logging, writing and reading from file.

"""
import os
import sys
import shutil
import glob
import csv
import pandas as pd
import logging

# For older versions of NEST we must call nest before calling mpi4py, otherwise nest.NumProcesses() gets set to 1.
import nest

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_nodes = comm.Get_size()
except:
    rank = 0
    n_nodes = 1

log_format = logging.Formatter('%(asctime)s [%(threadName)-12.12s] %(message)s')
pointnet_logger = logging.getLogger()
pointnet_logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
pointnet_logger.addHandler(console_handler)


def collect_gdf_files(gdf_dir, output_file, nest_id_map, overwrite=False):

    if n_nodes > 0:
        # Wait until all nodes are finished
        comm.Barrier()

    if rank != 0:
        return

    log("Saving spikes to file...")
    spikes_out = output_file
    if os.path.exists(spikes_out) and not overwrite:
        return

    gdf_files_globs = '{}/*.gdf'.format(gdf_dir)
    gdf_files = glob.glob(gdf_files_globs)
    with open(spikes_out, 'w') as spikes_file:
        csv_writer = csv.writer(spikes_file, delimiter=' ')
        for gdffile in gdf_files:
            spikes_df = pd.read_csv(gdffile, names=['gid', 'time', 'nan'], sep='\t')
            for _, row in spikes_df.iterrows():
                csv_writer.writerow([row['time'], nest_id_map[int(row['gid'])]])
            os.remove(gdffile)
    log("done.")


def setup_output_dir(config):
    if rank == 0:
        try:
            output_dir = config['output']['output_dir']
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            if 'log_file' in config['output']:
                file_logger = logging.FileHandler(config['output']['log_file'])
                file_logger.setFormatter(log_format)
                pointnet_logger.addHandler(file_logger)
                log('Created a log file')

        except Exception as exc:
            print(exc)

    try:
        comm.Barrier()
    except:
        pass


def quiet_nest():
    nest.set_verbosity('M_QUIET')


def log(message, all_ranks=False):
    if all_ranks is False and rank != 0:
        return

    # print message
    pointnet_logger.info(message)

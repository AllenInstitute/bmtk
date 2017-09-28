# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 20XX. Allen Institute. All rights reserved.
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
import glob
import csv
import pandas as pd

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_Rank()
except:
    rank = 0


def collect_gdf_files(gdf_dir, output_file, nest_id_map, overwrite=False):
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


def log(message, all_ranks=False):
    if all_ranks is False and rank != 0:
        return

    print(message)
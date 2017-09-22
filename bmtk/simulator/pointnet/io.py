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
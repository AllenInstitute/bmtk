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
"""
Functions for logging, writing and reading from file.

"""
import nest

from bmtk.simulator.core.io_tools import IOUtils

# Want users to be able to use NEST whether or not it is compiled in parallel mode or not, which means checking if
# the method nest.SyncPRocesses (aka MPI Barrier) exists. If it doesn't try getting barrier from mpi4py
rank = nest.Rank()
n_nodes = nest.NumProcesses()
try:
    barrier = nest.SyncProcesses
except AttributeError as exc:
    try:
        from mpi4py import MPI
        barrier = MPI.COMM_WORLD.Barrier
    except:
        # Barrier is just an empty function, no problem if running on one core.
        barrier = lambda: None


class NestIOUtils(IOUtils):
    def __init__(self):
        super(NestIOUtils, self).__init__()
        self.mpi_rank = rank
        self.mpi_size = n_nodes

    def barrier(self):
        barrier()

    def quiet_simulator(self):
        nest.set_verbosity('M_QUIET')

    def setup_output_dir(self, config_dir, log_file, overwrite=True):
        super(NestIOUtils, self).setup_output_dir(config_dir, log_file, overwrite=True)
        if n_nodes > 1 and rank == 0:
            io.log_info('Running NEST with MPI ({} cores)'.format(n_nodes))


io = NestIOUtils()


'''
log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
pointnet_logger = logging.getLogger()
pointnet_logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
pointnet_logger.addHandler(console_handler)


def collect_gdf_files(gdf_dir, output_file, nest_id_map, overwrite=False):

    if n_nodes > 0:
        # Wait until all nodes are finished
        barrier()

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

    barrier()
'''


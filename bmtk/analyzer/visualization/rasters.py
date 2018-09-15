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
import numpy as np


def plot_raster_query(ax, spikes, nodes_df, cmap, twindow=[0, 3], marker=".", lw=0, s=10):
    """Plot raster colored according to a query.

    Query's key defines node selection and the corresponding values defines color
    :param ax: matplotlib axes object, axes to use
    :param spikes: tuple of numpy arrays, includes [times, gids]
    :param nodes_df: pandas DataFrame, nodes table
    :param cmap: dict, key: query string, value:color
    :param twindow: tuple [start_time,end_time]
    :param marker:
    :param lw:
    :param s:
    """
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes[0] > tstart) & (spikes[0] < tend))

    spike_times = spikes[0][ix_t]
    spike_gids = spikes[1][ix_t]

    for query, col in cmap.items():
        query_df = nodes_df.query(query)
        gids_query = query_df.index
        print("{} ncells: {} {}".format(query, len(gids_query), col))

        ix_g = np.in1d(spike_gids, gids_query)
        ax.scatter(spike_times[ix_g], spike_gids[ix_g],
                   marker=marker,
                   # facecolors='none',
                   facecolors=col,
                   # edgecolors=col,
                   s=s,
                   label=query,
                   lw=lw)

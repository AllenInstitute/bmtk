import numpy as np



def plot_raster_query(ax,spikes,nodes_df,cmap,
                twindow=[0,3],
                marker=".",
                lw=0,
                s=10
                ):
    '''
    Plot raster colored according to a query. 
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays 
            includes [times, gids]
        nodes_df: pandas DataFrame
            nodes table
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]

    '''
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes[0]>tstart) & (spikes[0]<tend))

    spike_times = spikes[0][ix_t]
    spike_gids = spikes[1][ix_t]

    for query,col in cmap.items():
        query_df = nodes_df.query(query)
        gids_query = query_df.index
        print query,  "ncells:", len(gids_query), col

        ix_g = np.in1d(spike_gids, gids_query)
        ax.scatter(spike_times[ix_g],spike_gids[ix_g], 
                    marker= marker, 
                    #                     facecolors='none',
                    facecolors=col,
                    #                     edgecolors=col, 
                    s=s, 
                    label=query,
                    lw=lw
                    ); 
    



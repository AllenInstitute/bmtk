import matplotlib.pyplot as plt

from bmtk.simulator.filternet.lgnmodel.temporalfilter import TemporalFilterCosineBump, TemporalFilter
from bmtk.simulator.filternet.lgnmodel.spatialfilter import GaussianSpatialFilter
from bmtk.simulator.filternet.lgnmodel import movie

#################################################################################
#### Functions for generating some of the figures in the filternet notebook #####
#################################################################################


def plot_tfilter_params():
    gm = movie.GratingMovie(120, 240)
    mv = gm.create_movie(t_max=2.0)
    #
    # tf = TemporalFilterCosineBump(weights=[33.328, -20.10059],
    #                               kpeaks=[59.0, 120.0],  # [9.67, 20.03],
    #                               delays=[0.0, 0.0])
    #
    # tf.get_kernel(t_range=mv.t_range, threshold=0.0, reverse=True)
    # tf.imshow(t_range=mv.t_range, reverse=True)


    fig, axes = plt.subplots(3, 3, figsize=(10, 7))
    ri = ci = 0

    weights = [(30.0, -20.0), (30.0, -1.0), (15.0, -20.0)]
    kpeaks = [(3.0, 5.0), (3.0, 30.0), (20.0, 40.0)]
    delays = [(0.0, 0.0), (0.0, 60.0), (20.0, 60.0)]

    # weights control the amplitude of the peaks
    for ci, w in enumerate(weights):
        tf = TemporalFilterCosineBump(weights=w,
                                      kpeaks=[9.67, 20.03],
                                      delays=[0.0, 1.0])
        linear_kernel = tf.get_kernel(t_range=mv.t_range, reverse=True)
        axes[ri, ci].plot(linear_kernel.t_range[linear_kernel.t_inds], linear_kernel.kernel)
        axes[ri, ci].set_ylim([-3.5, 10.0])
        axes[ri, ci].text(0.05, 0.90, 'weights={}'.format(w), horizontalalignment='left', verticalalignment='top',
                          transform=axes[ri, ci].transAxes)

    axes[0, 0].set_ylabel('effect of weights')
    ri += 1

    # kpeaks parameters controll the spread of both peaks, the second peak must have a bigger spread
    for ci, kp in enumerate(kpeaks):
        tf = TemporalFilterCosineBump(weights=[30.0, -20.0],
                                      kpeaks=kp,
                                      delays=[0.0, 1.0])
        linear_kernel = tf.get_kernel(t_range=mv.t_range, reverse=True)
        axes[ri, ci].plot(linear_kernel.t_range[linear_kernel.t_inds], linear_kernel.kernel)
        axes[ri, ci].set_xlim([-0.15, 0.005])
        axes[ri, ci].text(0.05, 0.90, 'kpeaks={}'.format(kp), horizontalalignment='left', verticalalignment='top',
                          transform=axes[ri, ci].transAxes)
    axes[1, 0].set_ylabel('effects of kpeaks')
    ri += 1

    for ci, d in enumerate(delays):
        tf = TemporalFilterCosineBump(weights=[30.0, -20.0],
                                      kpeaks=[9.67, 20.03],
                                      delays=d)
        linear_kernel = tf.get_kernel(t_range=mv.t_range, reverse=True)
        axes[ri, ci].plot(linear_kernel.t_range[linear_kernel.t_inds], linear_kernel.kernel)
        axes[ri, ci].set_xlim([-0.125, 0.001])
        axes[ri, ci].text(0.05, 0.90, 'delays={}'.format(d), horizontalalignment='left', verticalalignment='top',
                          transform=axes[ri, ci].transAxes)

    axes[2, 0].set_ylabel('effects of delays')
    # plt.xlim()
    plt.show()


def plot_sfilter_params():
    gm = movie.GratingMovie(200, 200)
    mv = gm.create_movie(t_max=2.0)



    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    rotations = [0.0, 45.0]
    sigmas = [(30.0, 20.0), (20.0, 30.0)]

    for r, sigma in enumerate(sigmas):
        for c, rot in enumerate(rotations):
            gsf = GaussianSpatialFilter(translate=(0, 0), sigma=sigma, rotation=rot)
            axes[r, c].imshow(gsf.get_kernel(mv.row_range, mv.col_range).full(), extent=(0, 200, 0, 200), origin='lower')

            if r == 0:
                axes[r, c].title.set_text('spatial_rotation={}'.format(rot))

            if c == 0:
                axes[r, c].set_ylabel('spatial_size={}'.format(sigma))

    plt.show()


if __name__ == '__main__':
    # plot_tfilter_params()
    plot_sfilter_params()
    # tf.imshow(t_range=mv.t_range, reverse=True)
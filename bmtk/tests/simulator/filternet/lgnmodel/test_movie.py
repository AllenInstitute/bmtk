import pytest
import numpy as np
import matplotlib.pyplot as plt


from bmtk.simulator.filternet.lgnmodel import movie


def test_movie():
    mv = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))
    assert(mv.frame_rate == 1000.0)
    assert(mv.row_range.shape == (120,))
    assert(mv.row_range[0] == 0)
    assert(mv.row_range[-1] == 119)
    assert(mv.col_range.shape == (240,))
    assert(mv.col_range[0] == 0)
    assert(mv.col_range[-1] == 239)

    mv = movie.Movie(np.zeros((1001, 120, 240)), frame_rate=1000.0)
    assert(len(mv.t_range) == 1001)
    assert(mv.t_range[0] == 0.0)
    assert(mv.t_range[-1] == 1.0)


def test_add_movies():
    mv1 = movie.Movie(np.zeros((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))
    mv2 = movie.Movie(np.ones((1001, 120, 240)), t_range=np.linspace(0.0, 1.0, 1001, endpoint=True))
    mv3 = mv1 + mv2
    assert(mv3.data.shape == (2001, 120, 240))
    assert(len(mv3.t_range) == 2001)
    assert(np.sum(mv3[1000, :, :]) == 0)
    assert(np.sum(mv3[1001, :, :]) == 120*240)


def test_grating():
    gm = movie.GratingMovie(row_size=120, col_size=240, frame_rate=1000.0)

    # Default grating movie
    mv = gm.create_movie()
    assert(mv.data.shape == (1001, 60, 120))
    assert(mv.row_range[0] == 0.0)
    assert(mv.row_range[-1] == 120.0)
    assert(mv.col_range[0] == 0.0)
    assert(mv.col_range[-1] == 240.0)
    assert(mv.t_range[0] == 0)
    assert(mv.t_range[-1] == 1.0)

    # 2 second gratings with 0.5 second gray screen
    mv = gm.create_movie(t_max=2.0, gray_screen_dur=0.5)
    assert(mv.data.shape == (2001, 60, 120))
    assert(mv.t_range[0] == 0.0)
    assert(mv.t_range[-1] == 2.0)
    assert(np.sum(mv.data[0:500, :, :]) == 0) # check 0.5 second grey screen
    assert(np.sum(mv.data[501, :, :]) > 0)  # movie starts

    # disable aliasing
    mv = gm.create_movie(cpd=0.1)
    assert(mv.data.shape == (1001, 120, 240))
    assert(mv.t_range[0] == 0.0)
    assert(mv.t_range[-1] == 1.0)
    assert(mv.row_range[-1] == 120)
    assert(mv.col_range[-1] == 240)


if __name__ == '__main__':
    # test_movie()
    # test_add_movies()
    # test_grating()

    # gm = movie.GratingMovie(120, 240)
    # mv = gm.create_movie()
    # ffm = movie.FullFieldFlashMovie(range(60), range(80), 1.0, 2.0)
    # mv = ffm.full(t_max=2.0)
    lm = movie.LoomingMovie(120, 240)
    mv = lm.create_movie()

    mv.play()
    mv.imshow_summary()

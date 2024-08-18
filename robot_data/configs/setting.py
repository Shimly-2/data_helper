def init():
    global RESCALE, N_, M_, fps_, numcircles, mmpp, true_radius_mm, MASK_MARKERS_FLAG
    RESCALE = 1

    """
    N_, M_: the row and column of the marker array
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    ##  on GS mini with small dots. image size (h,w) (240,320)
    N_ = 7
    M_ = 9
    fps_ = 25

    numcircles = M_ * N_
    mmpp = .0625
    true_radius_mm = .5

    MASK_MARKERS_FLAG = True

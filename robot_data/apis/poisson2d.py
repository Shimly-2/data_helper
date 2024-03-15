import scipy
import numpy as np
import math

# 2D integration via Poisson solver
def poisson_reconstruct(grady, gradx, boundarysrc):
    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy
    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    #tt = np.zeros(boundarysrc.shape)
    #fsin = np.zeros(boundarysrc.shape)
    #cv2.dft(f, tt)
    #cv2.dft(tt.T, fsin)
    #cv2.namedWindow('dft')
    #cv2.imshow('dft',fsin)
    #cv2.waitKey()
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T
    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)
    f = fsin/denom
    # Inverse Discrete Sine Transform
    #img_tt = np.zeros(f.shape)
    #cv2.idft(f, tt)
    #cv2.idft(tt.T, img_tt)
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T
    # print('tt shape = ', tt.shape)
    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt
    return result
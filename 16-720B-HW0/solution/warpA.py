import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    y, x = np.mgrid[0:output_shape[0], 0:output_shape[1]]

    H, W = im.shape


    Ainv = np.linalg.inv(A)
    y_warped = np.round(Ainv[0, 0]*y + Ainv[0, 1]*x + Ainv[0, 2]).astype(int)
    x_warped = np.round(Ainv[1, 0]*y + Ainv[1, 1]*x + Ainv[1, 2]).astype(int)

    y_warped[(y_warped < 0) | (y_warped >= H)] = H
    x_warped[(x_warped < 0) | (x_warped >= W)] = W

    return np.pad(im, [[0, 1], [0, 1]], 'constant')[(y_warped, x_warped)]

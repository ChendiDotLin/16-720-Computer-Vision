import scipy.ndimage
import numpy as np


def warp(im, A, output_shape):
    return scipy.ndimage.affine_transform(im, np.linalg.inv(A),
                                          output_shape=output_shape, order=0)

import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""


    out_row = output_shape[0]
    out_col = output_shape[1]
    [im_row,im_col] = im.shape
    output = np.zeros((out_row,out_col))
    Ainv = np.linalg.inv(A);
    for i in range(out_row):
        for j in range(out_col):
            [ori_i,ori_j,temp] = np.matmul(Ainv,np.array([i,j,1]))
            ori_i = int(round(ori_i))
            ori_j = int(round(ori_j))
            if (ori_i >=0 and ori_i < im_row and ori_j >= 0 and ori_j < im_col):
                output[i][j] = im[ori_i][ori_j]
    return output

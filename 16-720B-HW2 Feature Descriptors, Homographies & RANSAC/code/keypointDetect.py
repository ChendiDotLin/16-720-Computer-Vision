import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt



def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    imH,imW,len_level = np.shape(gaussian_pyramid)
    DoG_pyramid = np.array([], dtype=np.int64).reshape(imH,imW,0)
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    for i in range(len(DoG_levels)):
        DoG_pyramid = np.dstack((DoG_pyramid,(gaussian_pyramid[:,:,i+1] - gaussian_pyramid[:,:,i])))
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ##################
    # TO DO ...
    # Compute principal curvature here
    imH,imW,len_level = np.shape(DoG_pyramid)
    principal_curvature = np.array([], dtype=np.int64).reshape(imH,imW,0)

    for i in range(len_level):
        dxx = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,2,0,ksize=3)
        dyy = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,0,2,ksize=3)
        dxy = cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,1,1,ksize=3)
        trace = dxx+dyy
        det = np.multiply(dxx,dyy) - np.multiply(dxy,dxy)
        if(0 in det ):
            print('beep')
            det[det==0] = 0.001
        R = np.divide(np.multiply(trace,trace),det)
        principal_curvature = np.dstack((principal_curvature,R))
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    imH,imW,len_level = np.shape(DoG_pyramid)
    neighborhood = generate_binary_structure(2,2)
    extremas = np.array([], dtype=np.int64).reshape(0, 3)
    # DoG_pyramid = abs(DoG_pyramid)
    for level in range(0,len_level):
        data_max = maximum_filter(DoG_pyramid[:,:,level],footprint = neighborhood,mode = 'constant')
        local_max = np.where((DoG_pyramid[:,:,level] == data_max) == True)
        dim,num_max = np.shape(local_max)
        local_max = np.concatenate((local_max[1].reshape(num_max,1),local_max[0].reshape(num_max,1),np.ones((num_max,1))*level),axis = 1)
        # print(np.shape(local_max))

        data_min = minimum_filter(DoG_pyramid[:,:,level],footprint = neighborhood,mode = 'constant')
        local_min = np.where((DoG_pyramid[:,:,level] == data_min) == True)
        dim,num_min = np.shape(local_min)
        local_min = np.concatenate((local_min[1].reshape(num_min,1),local_min[0].reshape(num_min,1),np.ones((num_min,1))*level),axis = 1)

        extrema = np.vstack((local_max,local_min))
        extremas = np.vstack((extremas,extrema))


    scale_min = argrelextrema(DoG_pyramid,np.less,axis = 2,mode = 'wrap')
    dim,num_scalemin = np.shape(scale_min)
    scale_min = np.concatenate((scale_min[1].reshape(num_scalemin,1),scale_min[0].reshape(num_scalemin,1),
                                scale_min[2].reshape(num_scalemin,1)),axis = 1)

    scale_max = argrelextrema(DoG_pyramid,np.greater,axis = 2,mode = 'wrap')
    dim, num_scalemax = np.shape(scale_max)
    scale_max= np.concatenate((scale_max[1].reshape(num_scalemax, 1), scale_max[0].reshape(num_scalemax, 1),
                                scale_max[2].reshape(num_scalemax, 1)), axis=1)
    scale = np.vstack((scale_min,scale_max))
    # print(contrast)
    # print(extremas)
    contrast = (np.where(abs(DoG_pyramid) > th_contrast))
    dim, num_contrast = np.shape(contrast)
    contrast = np.concatenate((contrast[1].reshape(num_contrast, 1), contrast[0].reshape(num_contrast, 1),
                               contrast[2].reshape(num_contrast, 1)), axis=1)

    r = np.where(abs(principal_curvature)<th_r)
    dim, num_r = np.shape(r)
    r = np.concatenate((r[1].reshape(num_r, 1), r[0].reshape(num_r, 1),
                               r[2].reshape(num_r, 1)), axis=1)
    # print(np.shape(r))
    # print (np.shape(contrast))
    # print(np.shape(scale))
    # print(np.shape(extremas))
    extremas_set = set([tuple(x) for x in extremas])
    scale_set = set([tuple(x) for x in scale])
    contrast_set = set([tuple(x) for x in contrast])
    r_set = set([tuple(x) for x in r])


    locsDoG = np.array([x for x in extremas_set & scale_set & contrast_set & r_set]).astype(int)

    # locsDoG = np.array([], dtype=np.int64).reshape(0, 3)
    # for level in range(1, len_level - 1):
    #     for i in range(1,imH-1):
    #         for j in range(1,imW-1):
    #             if extreme(DoG_pyramid[:,:,level],i,j): # extreme
    #                 if (((DoG_pyramid[i,j,level] < DoG_pyramid[i,j,level-1]) and (DoG_pyramid[i,j,level] < DoG_pyramid[i,j,level+1])) or
    #                     ((DoG_pyramid[i, j, level] > DoG_pyramid[i, j, level - 1]) and (DoG_pyramid[i, j, level] > DoG_pyramid[i, j, level + 1]))): # scale
    #                     if (abs(DoG_pyramid[i,j,level]) > th_contrast):
    #                         if (abs(principal_curvature[i,j,level])<th_r):
    #                             locsDoG = np.vstack((locsDoG,np.array([[j,i,level]])))

    # print (np.shape(locsDoG))
    ##############
    #  TO DO ...
    # Compute locsDoG here
    return locsDoG
    

def extreme(im,i,j):
    p = im[i,j]
    if (p < im[i-1,j-1] and p < im[i-1,j] and p<im[i-1,j+1] and p<im[i,j-1] and p<im[i,j+1] and p<im[i+1,j-1] and p<im[i+1,j] and p<im[i+1,j+1]):
        return True
    if (p > im[i-1,j-1] and p > im[i-1,j] and p>im[i-1,j+1] and p>im[i,j-1] and p>im[i,j+1] and p>im[i+1,j-1] and p>im[i+1,j] and p>im[i+1,j+1]):
        return True
    return False


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid,DoG_levels = createDoGPyramid(gauss_pyramid,levels)
    principle_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid,DoG_levels,principle_curvature,th_contrast,th_r)
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # print(locsDoG)
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fig = plt.imshow(im,cmap = 'gray')
    plt.plot(locsDoG[:,0],locsDoG[:,1],'o',color = 'lime',markersize = 3)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('edge_suppresion.png', bbox_inches='tight', pad_inches=0)

    plt.show()

import numpy as np
import LucasKanadeAffine
import InverseCompositionAffine
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)
    M = LucasKanadeAffine.LucasKanadeAffine(image1,image2)
    M = np.linalg.inv(M)
    # M = InverseCompositionAffine.InverseCompositionAffine(image1,image2)

    imH,imW = np.shape(image1)
    width = imW
    height = imH
    spline2 = RectBivariateSpline(np.linspace(0, imH, num=imH, endpoint=False),
                                  np.linspace(0, imW, num=imW, endpoint=False), image2)
    x, y = np.mgrid[0:imW, 0: imH]
    x = np.reshape(x, (1, height * width))
    y = np.reshape(y, (1, height * width))
    coor = np.vstack((x, y, np.ones((1, height * width))))
    coorp = np.matmul(M, coor)
    xp = coorp[0, :]
    yp = coorp[1, :]
    xp =np.reshape(xp, (width * height))
    yp =np.reshape(yp, (width * height))
    # warp_image1 = spline2.ev(yp, xp).reshape(height,width)
    warp_image1 = scipy.ndimage.affine_transform(image1,M[0:2,0:2],offset = M[0:2,2],output_shape = image2.shape)
    # print(np.shape(warp_image1))
    diff = abs(warp_image1 - image2)
    threshold = 0.2
    mask[diff > threshold] = 1
    mask[warp_image1==0.] = 0
    # print(mask)
    mask = scipy.ndimage.morphology.binary_erosion(mask,structure = np.ones((1,2)),iterations= 1)
    mask = scipy.ndimage.morphology.binary_erosion(mask,structure = np.ones((2,1)),iterations= 1)

    mask = scipy.ndimage.morphology.binary_dilation(mask,iterations = 1)

    # print(mask)
    return mask

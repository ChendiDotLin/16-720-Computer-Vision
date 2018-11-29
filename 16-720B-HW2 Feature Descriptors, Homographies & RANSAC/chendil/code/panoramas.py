import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    extra = 600
    imH,imW,channel = np.shape(im1)
    im1 = np.uint8(np.hstack((im1,np.zeros((imH,extra,channel)))))
    cv2.imshow('im1', im1)

    warp_im2 = np.uint8(cv2.warpPerspective(im2,H2to1,(imW+extra,imH)))
    cv2.imshow('warp2', warp_im2)
    # cv2.imshow('im1', im1)
    # warp.axes.get_xaxis().set_visible(False)
    # warp.axes.get_yaxis().set_visible(False)
    # pano_im = np.zeros((imH,imW+400,channel))
    pano_im = np.maximum(warp_im2,im1)
    pano_im = np.uint8(pano_im)
    cv2.imwrite('../results/q6_1.jpg', pano_im)

    return pano_im



def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    imH2,imW2,channel = np.shape(im2)
    imH1,imW1,channel = np.shape(im1)
    # y is the first row, x is the second row
    corner = np.array([[0,0,imW2-1,imW2-1],[0,imH2-1,imH2-1,0],[1,1,1,1]])
    # corner = np.array([[imW2-1],[imH2-1],[1]])

    trans_corner = np.matmul(H2to1,corner)
    trans_corner = (trans_corner/trans_corner[-1,:]).round().astype(int)
    print(trans_corner)
    Hmax = np.amax([imH1,np.amax(trans_corner[1,:])])
    Wmax = np.amax([imW1,np.amax(trans_corner[0,:])])
    Hmin = np.amin([0,np.amin(trans_corner[1,:])])
    Wmin = np.amin([0,np.amin(trans_corner[0,:])])
    totalH = Hmax - Hmin
    totalW = Wmax - Wmin
    print(Hmax,Wmax,Hmin,Wmin)
    #shift down = Hmin
    M = np.array([[1.0,0.0,0.0],[0.0,1.,-Hmin],[0.0,0.0,1.0]])
    print(totalH,totalW)
    print(M)
    print(np.shape(M))
    warp_im1 = cv2.warpPerspective(im1, M, (totalW,totalH))
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), (totalW,totalH))

    pano_im = np.maximum(warp_im2,warp_im1)

    # cv2.imshow('pano', pano_im)
    # cv2.waitKey(0)
    #
    pano_im = np.uint8(pano_im)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    return pano_im

def generatePanorama(im1,im2):


    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)


    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save('../results/q6_1.npy',H2to1)
    H2to1 = np.load('../results/q6_1.npy')
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    pano_im = imageStitching(im1,im2,H2to1)
    print(H2to1)
    generatePanorama(im1,im2)

    # h, status = cv2.findHomography(im2, im1) # used to verify h is correct
    # print(h)
    # cv2.imwrite('../results/panoImg.png', pano_im)
    # cv2.imshow('panoramas', pano_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
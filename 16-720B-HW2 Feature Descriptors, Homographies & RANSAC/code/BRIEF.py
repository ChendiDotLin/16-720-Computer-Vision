import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    '''
    #############################
    # TO DO ...
    # Generate testpattern here
    # compareXx = np.random.normal(0,1/5*(patch_width),nbits)
    # compareXx = compareXx.round().astype(int)
    # compareXy = np.random.normal(0, 1 / 5 * (patch_width), nbits)
    # compareXy = compareXy.round().astype(int)
    # compareYx = np.random.normal(0,1/5*(patch_width),nbits)
    # compareYx = compareYx.round().astype(int)
    # compareYy = np.random.normal(0, 1 / 5 * (patch_width), nbits)
    # compareYy = compareYy.round().astype(int)
    # compareX = (compareXx+4)*9 + compareXy+4
    # compareY = (compareYx+4)*9 + compareYy+4

    compareX = np.random.randint(patch_width*patch_width,size = nbits)
    compareY = np.random.randint(patch_width*patch_width,size = nbits)

    # print(np.shape(compareX))
    np.save('../results/testPattern.npy',np.vstack((compareX,compareY)))
    # print(np.load('testPattern.npy'))
    return  compareX, compareY

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################

    # TO DO ...
    # compute locs, desc here
    patch_width = 9

    imH,imW,len_level = np.shape(gaussian_pyramid)
    loc_num, col = np.shape(locsDoG)
    locs =  np.array([], dtype=np.int64).reshape(0, 3)
    n = np.shape(compareX)[0]
    desc =  np.array([], dtype=np.int64).reshape(0, n)
    for i in range(loc_num):
        locy = int(locsDoG[i,0])
        locx = int(locsDoG[i,1])
        layer = int(locsDoG[i,2])
        if (locx>=patch_width//2 and locx<=imH-(patch_width//2 + 1) and locy>= patch_width//2 and locy <= imW-(patch_width//2 +1)):
            locs = np.vstack((locs,locsDoG[i]))
            patch = gaussian_pyramid[locx - patch_width//2 : locx + patch_width//2 + 1, locy - patch_width//2:locy + patch_width//2 +1,layer]
            patch = patch.flatten()
            des = np.zeros((1,n))
            for k in range(len(compareX)):
                x = compareX[k]
                y = compareY[k]
                # if (x<=patch_width**2-1 and x>=0 and y<=patch_width**2-1 and y>=0):
                if (patch[x] < patch[y]):
                    des[0,k] = 1
            desc = np.vstack((desc,des))
    # print(np.shape(locs))
    # print(np.shape(desc))
    return locs, desc



def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    # TO DO ...
    locsDoG, gaussian_pyramid = DoGdetector(im)
    compare = np.load('../results/testPattern.npy')
    print(np.shape(compare))
    compareX = compare[0]
    compareY = compare[1]
    locs,desc = computeBrief(im, gaussian_pyramid, locsDoG, np.sqrt(2), [-1,0,1,2,3,4],
                 compareX, compareY)

    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    # test briefLite
    im = cv2.imread('../data/pf_scan_scaled.jpg')
    locs, desc = briefLite(im)

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
    # test matches
    # im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    im1 = cv2.imread('../data/chickenbroth_04.jpg')

    # im2 = cv2.imread('../data/pf_pile.jpg')
    im2 = cv2.imread('../data/model_chickenbroth.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    print('done')
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

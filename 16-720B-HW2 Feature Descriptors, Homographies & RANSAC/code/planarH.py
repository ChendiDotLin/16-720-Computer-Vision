import numpy as np
import cv2
import scipy
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    coor,N = np.shape(p1)
    u = p2[0,:].reshape(N,1)
    v = p2[1,:].reshape(N,1)
    x = p1[0,:].reshape(N,1)
    y = p1[1,:].reshape(N,1)
    A_up = np.hstack((-1*u,-1*v,-1*np.ones((N,1)),np.zeros((N,3)),np.multiply(x,u),np.multiply(x,v),x))
    A_down = np.hstack((np.zeros((N,3)),-1*u,-1*v,-1*np.ones((N,1)),np.multiply(y,u),np.multiply(y,v),y))
    A = np.vstack((A_up,A_down))
    u,s,vh = np.linalg.svd(A)
    # print(u,s,vh)
    # print(scipy.linalg.null_space(A))
    h = vh[-1,:]
    # print (h)
    # h, status = cv2.findHomography(p2.T, p1.T)
    # print(h)
    H2to1 = h.reshape(3,3)
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    print(matches)
    match_locs1 = locs1[matches[:,0]]
    match_locs1 = match_locs1[:,[0,1]]
    match_locs2 = locs2[matches[:,1]]
    match_locs2 = match_locs2[:,[0,1]]
    N = np.shape(match_locs1)[0]

    homo_match1 = np.vstack((match_locs1.T,np.ones((1,N))))
    homo_match2 = np.vstack((match_locs2.T,np.ones((1,N))))

    num_inlier = 0
    print(N)
    for i in range(num_iter):
        idx = np.random.choice(N, 4)
        # print(idx)
        p1 = match_locs1[idx]
        p1 = p1.T
        p2 = match_locs2[idx]
        p2 = p2.T
        H = computeH(p1,p2)
        # apply H to homo match 2
        trans_match2 = np.matmul(H,homo_match2)
        lam = trans_match2[-1, :]
        trans_match2 = trans_match2/lam
        # compare transmatch 2 to homo match 1
        diff = (trans_match2 - homo_match1)[[0,1],:]
        norm_diff = np.linalg.norm(diff,axis = 0)
        inlier_now = np.where(norm_diff < tol)

        if (np.shape(inlier_now)[1] > num_inlier):
            num_inlier = np.shape(inlier_now)[1]
            inlier = inlier_now
        # p1ho = np.vstack((p1,np.ones((1,4))))
        # p1tran = np.matmul(H,p1ho)
        # lam = p1tran[-1, :]
        # p1tran = p1tran/lam
        # print(p1tran[[0,1],:]-p2)
        # print(p2)
    print('num_inliner',num_inlier)
    p1 = match_locs1[inlier]
    p1 = p1.T
    p2 = match_locs2[inlier]
    p2 = p2.T
    bestH = computeH(p1,p2)
    # trans_match1 = np.matmul(bestH, homo_match1)
    # lam = trans_match1[-1, :]
    # trans_match1 = trans_match1 / lam
    # # compare transmatch 1 to homo match 2
    # diff = (trans_match1 - homo_match2)[[0, 1], :]
    # norm_diff = np.linalg.norm(diff, axis=0)
    # print(norm_diff)
    # pass
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)


import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
import BRIEF
import matplotlib.pyplot as plt

if __name__ == '__main__':
    im = cv2.imread('../data/model_chickenbroth.jpg')
    rows,cols,channel = np.shape(im)
    data = []
    locs1, desc1 = BRIEF.briefLite(im)
    correct = []
    for i in range(36):
        angle = 10*(i)
        rot = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        print(np.shape(rot))
        dst = cv2.warpAffine(im,rot,(cols,rows))
        # cv2.imshow('test',dst)
        # cv2.waitKey(0)  # press any key to exit
        locs2, desc2 = BRIEF.briefLite(dst)
        matches = BRIEF.briefMatch(desc1, desc2)

        match_locs1 = locs1[matches[:, 0]]
        match_locs1 = match_locs1[:, [0, 1]].T
        match_locs2 = locs2[matches[:, 1]]
        match_locs2 = match_locs2[:, [0, 1]]

        coor,m = np.shape(match_locs1)
        ori = np.vstack((match_locs1,np.ones((1,m))))
        trans =rot.dot(ori).T


        diff = np.linalg.norm(trans - match_locs2,axis = 1)
        well = diff[diff<10]

        # print(diff,well)
        correct.append(len(well))

        data.append(len(matches))
    print(data)
    print(np.shape(data))
    print(correct)
    # data = np.reshape(data,(1,len(data)))
    bin = np.linspace(0,350,num = 36,endpoint=True)
    plt.bar(bin, correct, width=1)
    plt.xlim(min(bin)-5, max(bin)+5)
    plt.show()



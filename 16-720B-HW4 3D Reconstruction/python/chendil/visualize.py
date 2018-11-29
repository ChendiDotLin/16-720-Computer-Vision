'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path
import submission as sub
import helper
import findM2



def q42 ():
    M1, C1, M2, C2, F = findM2.findM2()
    data = np.load('../data/templeCoords.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    x1 = data['x1']
    y1 = data['y1']
    print(x1.shape)
    num, temp = x1.shape
    pts1 = np.hstack((x1, y1))
    pts2 = np.zeros((num, 2))
    for i in range(num):
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i, 0], y1[i, 0])
        pts2[i, 0] = x2
        pts2[i, 1] = y2
        print(i)


    P, err = sub.triangulate(C1, pts1, C2, pts2)
    # print(err)
    # bundle adjustment
    # M2, P = sub.bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim3d(3.4, 4.2)
    ax.set_xlim3d(-0.8, 0.6)


    plt.show()
    if (os.path.isfile('q4_2.npz') == False):
        np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)


def q53():
    data = np.load('../data/some_corresp_noisy.npz')

    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    # print(np.shape(im1))
    # print(M)

    # using RANSAC to find F
    F,inliers = sub.ransacF(data['pts1'], data['pts2'], M)

    E = sub.essentialMatrix(F, K1, K2)
    # print(E)
    M1 = np.hstack(((np.eye(3)), np.zeros((3, 1))))
    M2s = helper.camera2(E)
    row, col, num = np.shape(M2s)
    # print(M1)
    C1 = np.matmul(K1, M1)
    # minerr = np.inf
    # res = 0
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = sub.triangulate(C1, pts1[inliers], C2, pts2[inliers])
        if (np.all(P[:, 2] > 0)):
            break
    # P, err = sub.triangulate(C1, pts1[inliers], C2, pts2[inliers])
    M2, P = sub.bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], P)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_zlim3d(3., 4.5)

    plt.show()


if __name__ == '__main__':
    q42()
    q53()

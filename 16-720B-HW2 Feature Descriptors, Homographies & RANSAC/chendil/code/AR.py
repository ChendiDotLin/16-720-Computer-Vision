import numpy as np
import planarH
import re
import matplotlib.pyplot as plt
import skimage.io


def compute_extrinsics (K,H):
    # a = np.array([[1,2,3],[3,4,5],[5,4,3]])
    # print(a)
    # print(a[:,[0,2]])
    # print(a[:,1].reshape(3,1))
    # print (np.hstack([a[:,[0,2]],a[:,1].reshape(3,1)]))
    phip = np.matmul(np.linalg.inv(K),H)
    # print(phip)
    u,l,vh = np.linalg.svd(phip[:,[0,1]])
    rotate_firsttwo = np.matmul(u,(np.matmul(np.array([[1,0],[0,1],[0,0]]),vh)))
    third = np.cross(rotate_firsttwo[:,0],rotate_firsttwo[:,1])
    R = np.hstack([rotate_firsttwo,third.reshape(3,1)])
    if (np.linalg.det(R) == -1):
        R[:,2] *= -1
    # print(R)
    factor = 0
    for m in range(3):
        for n in range(2):
            factor += phip[m,n]/rotate_firsttwo[m,n]
    # print(factor)
    factor /= 6
    t = phip[:,2]/factor
    print(np.matmul(R,R.T))
    return R,t.reshape(3,1)


def project_extrinsics(K,W,R,t):
    print(np.shape(np.matmul(R,W)))
    print(np.shape(t))
    X = np.matmul(R,W)

    X= X+t
    X = np.matmul(K,X)
    lam = X[-1, :]
    X = X / lam
    # print(np.shape(lam))
    print(X)
    # print(np.shape(X))
    return(X)

if __name__ == '__main__':
    X2 = np.array([[0,18.2,18.2,0],[0,0,26,26]])
    X1 = np.array([[483,1704,2175,67],[810,781,2217,2286]])
    H = planarH.computeH(X1,X2)
    K = np.array([[3043.72,0.0,1196.0],[0.0,3043.72,1604.0],[0.0,0.0,1.0]])
    print(K)
    R,t = compute_extrinsics(K,H)
    textfile = open('../data/sphere.txt','r')
    text = textfile.readlines()
    W = np.array([], dtype=np.int64).reshape(0, 961)
    print(np.shape(text))
    for line in text:
        # coor = re.findall(r"[-+]?\d*\.\d+|d+",line)
        # coor = list(map(float,coor))
        # coor = np.array(coor)
        number_str = line.split()
        coor = np.array([float(x) for x in number_str])
        coor = coor.reshape(1,np.shape(coor)[0])
        W = np.vstack((W,coor))
        # print(W)
        # print(np.shape(W))

    # W = W[[0,1],:]
    # W = np.vstack((W,np.ones((1,961))))
    textfile.close()
    offset = np.array([[5],[10],[3]])
    W = W+offset
    print(W[:,1])
    print(R,t)
    # W = np.array([[0,18.2,18.2,0],[0,0,26,26],[0,0,0,0]])
    # W[2,:] = np.ones((1,961))
    X = project_extrinsics(K,W,R,t)
    im = skimage.io.imread('../data/prince_book.jpeg')
    fig = plt.imshow(im)
    X2trans = project_extrinsics(K,np.array([[0,18.2,18.2,0],[0,0,26,26],[0,0,0,0]]),R,t)
    plt.plot(X2trans[0, :], X2trans[1,:], 'o', color='red', markersize=3)

    plt.plot(X[0, :], X[1,:], '.', color='yellow', markersize=3)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


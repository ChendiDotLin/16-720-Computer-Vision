import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    p0 = np.zeros(6)
    imH0, imW0 = np.shape(It)
    imH1, imW1 = np.shape(It1)
    # print(np.shape(It))

    width = imW0
    height = imH0
    # print(It[imH0-1,imW0-1])
    spline0 = RectBivariateSpline(np.linspace(0, imH0, num=imH0, endpoint=False),
                                  np.linspace(0, imW0, num=imW0, endpoint=False), It)
    spline1 = RectBivariateSpline(np.linspace(0, imH1, num=imH1, endpoint=False),
                                  np.linspace(0, imW1, num=imW1, endpoint=False), It1)

    p = p0
    # print(p)
    threshold = 0.01
    change = 1
    counter = 1
    x, y = np.mgrid[0:imW0, 0:imH0]
    # print(np.shape(x))
    x = np.reshape(x, (1, height * width))
    y = np.reshape(y, (1, height * width))
    coor = np.vstack((x, y, np.ones((1, height * width))))
    dxT = spline0.ev(y, x, dy=1).flatten()
    dyT = spline0.ev(y, x, dx=1).flatten()
    Itp = spline0.ev(y, x).flatten()
    x = np.reshape(x, (height * width, 1))
    y = np.reshape(y, (height * width, 1))
    dxT = np.reshape(dxT, (height * width, 1))
    dyT = np.reshape(dyT, (height * width, 1))
    A1 = np.multiply(x, dxT)
    A2 = np.multiply(y, dxT)
    A4 = np.multiply(x, dyT)
    A5 = np.multiply(y, dyT)
    # print('A5 done')
    Ap = np.hstack((A1, A2, dxT, A4, A5, dyT))
    precompute = np.matmul(np.linalg.pinv(np.matmul(Ap.T,Ap)),Ap.T)
    while (change > threshold and counter < 50):
        M = np.array([[1 + p[0], p[1], p[2]],
                      [p[3], 1 + p[4], p[5]],
                      [0, 0, 1]])
        coorp = np.matmul(M, coor)
        xp = coorp[0]
        yp = coorp[1]

        It1p = spline1.ev(yp, xp).flatten()

        b = np.reshape(Itp - It1p, (len(xp), 1))
        deltap = precompute.dot(b)

        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        # print(p)
        counter += 1
    M = np.array([[1 + p[0], p[1], p[2]],
                  [p[3], 1 + p[4], p[5]],
                  [0, 0, 1]])
    return M

import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    # M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
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
    x, y = np.mgrid[0:imW0,0:imH0]
    # print(np.shape(x))
    x = np.reshape(x,(1,height*width))
    y = np.reshape(y,(1,height*width))
    coor = np.vstack((x, y, np.ones((1, height * width))))

    while (change > threshold and counter < 50):
        M = np.array([[1+p[0], p[1],p[2]],
                      [p[3],1+p[4],p[5]],
                      [0,0,1]])
        coorp = np.matmul(M,coor)
        xp = coorp[0]
        yp = coorp[1]
        xout = (np.where(xp>=imW0) or np.where(xp < 0))

        yout = (np.where(yp>=imH0) or np.where(yp < 0))
        # print(M)
        # print(np.shape(xout))
        # print(xout,yout)
        if (np.shape(xout)[1] == 0 and np.shape(yout)[1] == 0):
            out = []
        elif (np.shape(xout)[1] != 0 and np.shape(yout)[1] ==0):
            # print(xout)
            out = xout
        # print(np.shape(xout))
        # print(xout)
        # print(np.shape(yout))
        elif (np.shape(xout)[1] == 0 and np.shape(yout)[1] !=0):
            out = yout
        else:

            out = np.unique(np.concatenate((xout,yout),0))

        xnew = np.delete(x,out)
        ynew = np.delete(y,out)
        xp = np.delete(xp,out)
        yp = np.delete(yp,out)
        dxp = spline1.ev(yp, xp, dy=1).flatten()
        dyp = spline1.ev(yp, xp, dx=1).flatten()
        It1p = spline1.ev(yp, xp).flatten()
        Itp = spline0.ev(ynew, xnew).flatten()

        xnew =  np.reshape(xnew,(len(xnew),1))
        ynew =  np.reshape(ynew,(len(ynew),1))
        xp = np.reshape(xp,(len(xp),1))
        yp = np.reshape(yp,(len(yp),1))
        dxp = np.reshape(dxp, (len(dxp), 1))
        dyp = np.reshape(dyp, (len(dyp), 1))

        A1 = np.multiply(xnew,dxp)

        A2 = np.multiply(ynew,dxp)
        A4 = np.multiply(xnew,dyp)
        A5 = np.multiply(ynew,dyp)
        A = np.hstack((A1,A2,dxp,A4,A5,dyp))

        # print(np.shape(A))
        b = np.reshape(Itp - It1p, (len(xp), 1))
        deltap = np.linalg.pinv(A).dot(b)
        # deltap = np.linalg.lstsq(A,b,rcond = -1)[0]
        # print(deltap)
        # print(np.shape(deltap))
        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        # print(p)
        counter += 1
        # print(counter)
    M = np.array([[1+p[0], p[1],p[2]],
                      [p[3],1+p[4],p[5]],
                      [0,0,1]])
    return M

import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
import cv2
from scipy import signal

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    imH0,imW0 = np.shape(It)
    imH1,imW1 = np.shape(It1)


    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]


    width = int(x2-x1)
    height = int(y2-y1)
    # print(It[imH0-1,imW0-1])
    spline0 = RectBivariateSpline(np.linspace(0,imH0,num=imH0,endpoint=False),np.linspace(0,imW0,num=imW0,endpoint=False),It)
    spline1 = RectBivariateSpline(np.linspace(0,imH1,num=imH1,endpoint=False),np.linspace(0,imW1,num=imW1,endpoint=False),It1)


    p = p0
    # print(p)
    threshold = 0.01
    change = 1
    counter = 1
    x,y = np.mgrid[x1:x2+1:width*1j,y1:y2+1:height*1j]
    while (change > threshold and counter < 50):

        dxp = spline1.ev(y+p[1], x+p[0],dy = 1).flatten()
        dyp = spline1.ev(y+p[1], x+p[0],dx = 1).flatten()
        It1p = spline1.ev(y+p[1], x+p[0]).flatten()
        Itp = spline0.ev(y, x).flatten()


        A = np.zeros((width*height,2*width*height))
        for i in range(width*height):
            # print(dxp)
            # print(dxp[i])
            A[i,2*i] = dxp[i]
            A[i,2*i+1] = dyp[i]
        # # print(A)
        A = np.matmul(A,(np.matlib.repmat(np.eye(2),width*height,1)))
        # A =np.hstack((dxp.reshape((width*height,1)),dyp.reshape((width*height,1))))
        # print(A)
        # print(np.shape(A))
        b = np.reshape(Itp - It1p,(width*height,1))
        deltap = np.linalg.pinv(A).dot(b)
        # deltap = np.linalg.lstsq(A,b,rcond = -1)[0]
        # print(deltap)
        # print(np.shape(deltap))
        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        # print(p)
        counter+=1
    return p


import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    baseH, baseW, basenum = np.shape(bases)
    # print(baseH, baseW, basenum)
    imH0, imW0 = np.shape(It)
    imH1, imW1 = np.shape(It1)

    # x1 =float('%.9f'%(rect[0]))
    # y1 =float('%.9f'%(rect[1]))
    # x2 =float('%.9f'%(rect[2]))
    # y2 =float('%.9f'%(rect[3]))

    # print(x1)
    x1 = (rect[0])
    y1 = (rect[1])
    x2 = (rect[2])
    y2 = (rect[3])
    # print(x1)
    spline0 = RectBivariateSpline(np.linspace(0, imH0, num=imH0, endpoint=False),
                                  np.linspace(0, imW0, num=imW0, endpoint=False), It)
    spline1 = RectBivariateSpline(np.linspace(0, imH1, num=imH1, endpoint=False),
                                  np.linspace(0, imW1, num=imW1, endpoint=False), It1)

    p = np.zeros(2)
    # print(p)
    threshold = 0.01
    change = 1
    counter = 1
    x, y = np.mgrid[x1:x2 + 1.:baseW*1j, y1:y2 + 1.:baseH*1j]
    height = np.shape(x)[1]
    width = np.shape(x)[0]
    It1 = spline1.ev(y, x)
    It = spline0.ev(y, x)
    # print(x1,x2,y1,y2)
    # print(np.shape(It))
    # print(np.shape(It1))
    # print(np.shape(It))
    w = []
    B = np.array([], dtype=np.int64).reshape(baseW*baseH, 0)

    for i in range(basenum):
        base = bases[:,:,i].T
        # print(np.shape(base))
        w.append((np.matmul((base.T),It1-It))/(np.matmul((base.T),(base))))
        B = np.hstack((B,base.reshape(baseW*baseH,1)))
    # B = np.sum(bases,axis = 2).T
    # print(np.shape(B))
    while (change > threshold):

        dxp = spline1.ev(y+p[1], x+p[0],dy = 1).flatten()
        dyp = spline1.ev(y+p[1], x+p[0],dx = 1).flatten()
        It1p = spline1.ev(y+p[1], x+p[0]).flatten()
        Itp = It.flatten()

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
        # print(b)
        # deltap = np.linalg.lstsq(A-np.matmul(np.matmul(B,B.T),A),b-np.matmul(np.matmul(B,(B.T)),b),rcond = -1)[0]
        deltap = np.linalg.pinv(A-np.matmul(np.matmul(B,B.T),A)).dot(b-np.matmul(np.matmul(B,(B.T)),b))

        # print(deltap)
        # print(np.shape(deltap))
        change = np.linalg.norm(deltap)
        p = (p + deltap.T).ravel()
        # print(p)
        counter+=1
    return p


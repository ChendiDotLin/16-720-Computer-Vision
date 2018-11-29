"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import sympy
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import os.path

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1*1.0/M
    pts2 = pts2*1.0/M
    n,temp = pts1.shape
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    A1 = (x2*x1)
    # print(A1.shape)
    A2 = (x2*y1)
    A3 = x2
    A4 = y2*x1
    A5 = y2*y1
    A6 = y2
    A7 = x1
    A8 = y1
    A9 = np.ones(n)
    A = np.vstack((A1,A2,A3,A4,A5,A6,A7,A8,A9))
    A = A.T
    u, s, vh = np.linalg.svd(A)
    f = vh[-1, :]
    F = f.reshape(3,3)
    F = helper._singularize(F)
    F = helper.refineF(F,pts1,pts2)
    T = np.array([[1./M,0,0],[0,1./M,0],[0,0,1]])
    F = np.matmul(T.T,np.matmul(F,T))
    # print(F)
    if(os.path.isfile('q2_1.npz')==False):
        np.savez('q2_1.npz',F = F, M = M)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1 * 1.0 / M
    pts2 = pts2 * 1.0 / M
    n, temp = pts1.shape
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]
    A1 = (x2 * x1)
    A2 = (x2 * y1)
    A3 = x2
    A4 = y2 * x1
    A5 = y2 * y1
    A6 = y2
    A7 = x1
    A8 = y1
    A9 = np.ones(n)
    A = np.vstack((A1, A2, A3, A4, A5, A6, A7, A8, A9))
    A = A.T
    u, s, vh = np.linalg.svd(A)
    f1 = vh[-1, :]
    f2 = vh[-2, :]
    F1 = f1.reshape(3,3)
    F2 = f2.reshape(3,3)
    F1 = helper.refineF(F1,pts1,pts2)
    F2 = helper.refineF(F2,pts1,pts2)

    # w = sympy.Symbol('w')
    # m = sympy.Matrix(w*F1+(1-w)*F2)
    #
    # coeff = (m.det().as_poly().coeffs())
    fun = lambda a: np.linalg.det(a*F1+(1-a)*F2)
    a0 = fun(0)
    a1 = (fun(1) - fun(-1))/3-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
    a3 = (fun(1) - fun(-1))/6 + (fun(2) - fun(-2))/12

    coeff = [a3,a2,a1,a0]
    # print(coeff)
    # coeff = coeff[::-1]
    # print(coeff)
    soln = np.roots(coeff)
    soln = soln[np.isreal(soln)]
    # print(soln)
    # print(soln.shape)
    Fs = []
    T = np.array([[1./ M, 0, 0], [0, 1./ M, 0], [0, 0, 1]])
    for i in range(len(soln)):
        F = (np.matmul(T.T, np.matmul((soln[i]*F1+(1-soln[i])*F2), T)))
        # F = helper.refineF(F,pts1,pts2)
        # F = helper._singularize(F)
        Fs.append(F)
    # print (Fs)
    if(os.path.isfile('q2_2.npz')==False):
        np.savez('q2_2.npz', F = Fs[2], M = M, pts1 = pts1, pts2 = pts2)
    return Fs

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.matmul(K2.T,np.matmul(F,K1))
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    n, temp = pts1.shape
    P = np.zeros((n,3))
    Phomo = np.zeros((n,4))
    for i in range(n):
        x1 = pts1[i,0]
        y1 = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]
        A1 = x1*C1[2,:] - C1[0,:]
        A2 = y1*C1[2,:] - C1[1,:]
        A3 = x2*C2[2,:] - C2[0,:]
        A4 = y2*C2[2,:] - C2[1,:]
        A = np.vstack((A1,A2,A3,A4))
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        p = p/p[3]
        P[i, :] = p[0:3]
        Phomo[i, :] = p
        # print(p)
    p1_proj = np.matmul(C1,Phomo.T)
    lam1 = p1_proj[-1,:]
    p1_proj = p1_proj/lam1
    p2_proj = np.matmul(C2,Phomo.T)
    lam2 = p2_proj[-1,:]
    p2_proj = p2_proj/lam2
    err1 = np.sum((p1_proj[[0,1],:].T-pts1)**2)
    err2 = np.sum((p2_proj[[0,1],:].T-pts2)**2)
    err = err1 + err2
    print(n)
    print(err)

    return P,err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pt1 = np.array([[x1],[y1],[1]])
    data = np.load('../data/some_corresp.npz')
    epiline = F.dot(pt1)
    # print(epiline.shape)
    a = epiline[0]
    b = epiline[1]
    c = epiline[2]
    # ax+by+c = 0
    # print(a,b,c)
    H,W,channel = im1.shape
    liney = np.arange(y1-30,y1+30)
    # print(liney.shape)
    # print(liney.shape)
    linex = (-(b*liney+c)/a)
    window = 5
    # print(x1,y1)
    im1g = ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    im2g = ndimage.gaussian_filter(im2, sigma=1, output=np.float64)
    # im1g = im1
    # im2g = im2
    # print(im1g.shape)
    minerr = np.inf
    res = 0
    for i in range(60):
        x2 = int(linex[i])
        y2 = liney[i]
        # print(x2,y2)
        if (x2>=window  and x2<= W-window-1 and y2>=window and y2<= H-window-1):
            patch1 = im1g[y1-window:y1+window+1, x1-window:x1+window+1,:]
            patch2 = im2g[y2-window:y2+window+1, x2-window:x2+window+1,:]
            diff = (patch1-patch2).flatten()
            err= (np.sum(diff**2))
            if (err<minerr):
                minerr = err
                res = i
    if (os.path.isfile('q4_1.npz') == False):
        np.savez('q4_1.npz', F=F, pts1=data['pts1'], pts2=data['pts2'])
    return linex[res],liney[res]

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    # error matrics is x2^TFx1 = 0
    num_iter = 200
    num_inlier = 0

    n, temp = pts1.shape
    threshold = 2e-3
    for i in range(num_iter):
        print(i)
        idx = np.random.choice(n,7)
        chosen_pts1 = pts1[idx,:]
        chosen_pts2 = pts2[idx,:]
        Fs = sevenpoint(chosen_pts1, chosen_pts2, M)
        # print(len(Fs))
        for j in range(len(Fs)):
            F = Fs[j]
            inliers_now = []
            for k in range(n):
                x2 = np.append(pts2[k,:],1).reshape((1,3))
                x1 = np.append(pts1[k,:],1).reshape((3,1))
                err = abs(np.matmul(x2,np.matmul(F,x1)))
                if (err<threshold):
                    inliers_now.append(k)
            if(len(inliers_now) > num_inlier):
                num_inlier = len(inliers_now)
                inliers = inliers_now
        print(num_inlier)
    inliers1 = pts1[inliers,:]

    inliers2 = pts2[inliers,:]

    F = eightpoint(inliers1, inliers2, M)
    # NOTE HERE! it will write a new q2_1.npz to replace the previous one
    print(num_inlier)

    return F,inliers

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    t = np.linalg.norm(r)
    if (t == 0):
        return np.eye(3)
    u = r/t
    u1 = u[0,0]
    u2 = u[1,0]
    u3 = u[2,0]
    ucross = np.array([[0, -u3, u2],
                       [u3, 0, -u1],
                       [-u2, u1, 0]])
    R = np.eye(3)*np.cos(t)+(1-np.cos(t))*u.dot(u.T)+np.sin(t)*ucross
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    rho = np.array([[A[2,1]],[A[0,2]],[A[1,0]]])
    s = np.linalg.norm(rho)
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    if (s == 0 and c == 1):
        return np.zeros((3,1))
    if (s == 0 and c == -1):
        temp = R + np.eye(3)
        if (np.linalg.norm(temp[:,0]) != 0):
            v = temp[:,0]
        elif(np.linalg.norm(temp[:,1]) != 0):
            v = temp[:,1]
        else:
            v = temp[:,2]
        u = v/np.linalg.norm(v)
        r = u*np.pi
        r1 = r[0,0]
        r2 = r[1,0]
        r3 = r[2,0]
        if ((r1== 0 and r2== 0 and r3<0) or (r1 == 0 and r2<0) or (r1<0)):
            r = -1*r
        return r
    if (s != 0):
        u = rho/s
        t = np.arctan2(s,c)
        r = t*u
        return r






'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    C1 = np.matmul(K1, M1)
    # print(x.shape)
    n,temp = p1.shape
    P = x[0:3*n].reshape((n,3))
    Phomo = np.hstack((P,np.ones((n,1))))
    # print(Phomo.shape)
    r2 = x[3*n:3*n+3].reshape((3,1))
    t2 = x[3*n+3:].reshape((3,1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2))
    C2 = np.matmul(K2,M2)
    p1_proj = np.matmul(C1, Phomo.T)
    lam1 = p1_proj[-1, :]
    p1_proj = p1_proj / lam1
    p2_proj = np.matmul(C2, Phomo.T)
    lam2 = p2_proj[-1, :]
    p2_proj = p2_proj / lam2
    p1_hat = p1_proj[0:2,:].T
    p2_hat = p2_proj[0:2,:].T
    residuals = np.concatenate([(p1 - p1_hat).reshape([-1]), (p2 - p2_hat).reshape([-1])]).reshape(4*n,1)
    # print(residuals.shape)

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    funcWrapper = lambda x: (rodriguesResidual(K1,M1,p1,K2,p2,x).flatten())
    x0 = P_init.flatten()
    print(x0.shape)
    R2_init = M2_init[:,0:3]
    r2_init = invRodrigues(R2_init).flatten()
    t2_init = M2_init[:,3].flatten()
    n,temp = p1.shape
    x0 = np.hstack((x0, r2_init,t2_init))
    x,ier = optimize.leastsq(funcWrapper, x0)
    print(np.shape(x))
    print(ier)
    print(np.sum(funcWrapper(x)**2))
    P2 = x[0:3 * n].reshape((n, 3))
    r2 = x[3*n:3*n+3].reshape((3, 1))
    t2 = x[3*n+3:].reshape((3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))
    return M2, P2





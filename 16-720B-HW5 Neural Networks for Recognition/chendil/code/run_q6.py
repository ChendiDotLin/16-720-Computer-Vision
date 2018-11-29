import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
u,s,vh = np.linalg.svd(train_x)
print(u.shape,s.shape,vh.shape)
sarray = np.array(s)
sort_index = np.argsort(sarray)
sort_index = sort_index[::-1]
# u_pca = u[:,sort_index[0:dim]]
# s_pca = s[sort_index[0:dim]]
vh_pca = vh[sort_index[0:dim],:]
# print(s[sort_index[0:dim]])
print(vh_pca.shape)

# rebuild a low-rank version
# lrank = None
P = vh_pca.T
lrank = np.matmul(train_x,P)
# P should be (1024,32) D = 1024, D' = 32, Y = XP
# rebuild it
# recon = None
recon = np.matmul(lrank,P.T)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
# recon_valid = None
recon_valid = np.matmul(np.matmul(valid_x,P),P.T)




for i in range(5):
    plt.subplot(2,1,1)
    index = int(3600/5*i)
    plt.imshow(valid_x[index,:].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon_valid[index,:].reshape(32,32).T)
    plt.show()
    plt.subplot(2, 1, 1)
    index = int(3600 / 5 * i + 1)
    plt.imshow(valid_x[index, :].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(recon_valid[index, :].reshape(32, 32).T)
    plt.show()

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())


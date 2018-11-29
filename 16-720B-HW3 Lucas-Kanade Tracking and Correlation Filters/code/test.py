import numpy as np

a = np.load('code/carseqrects.npy')
aa = np.load('data/carseq.npy')
print(a.shape,aa.shape)


b = np.load('code/carseqrects-wcrt.npy')
bb = np.load('data/carseq.npy')
print(b.shape,bb.shape)


c = np.load('code/sylvseqrects.npy')
cc = np.load('data/sylvseq.npy')
print(c.shape,cc.shape)
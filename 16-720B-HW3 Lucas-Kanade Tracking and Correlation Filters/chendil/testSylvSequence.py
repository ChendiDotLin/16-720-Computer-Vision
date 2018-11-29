import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis
# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    sylvseq = np.load('../data/sylvseq.npy')
    imH, imW, frames = np.shape(sylvseq)
    print(frames)
    bases = np.load('../data/sylvbases.npy')

    rect = [101., 61., 155., 107.]
    rects = rect[:]
    for frame in range(frames - 1):

        print(frame)
        It = sylvseq[:, :, frame]
        It1 = sylvseq[:, :, frame + 1]
        p = LucasKanadeBasis.LucasKanadeBasis(It, It1, rect,bases)
        rect[0] += p[0]
        rect[2] += p[0]
        rect[1] += p[1]
        rect[3] += p[1]
        # print(rect)
        rects = np.vstack((rects, rect))
    #     # print(rects)
    #     # plt.imshow(It1, cmap='gray')
    #     # patch = patches.Rectangle((rect[0], rect[1]), (rect[2] - rect[0]), (rect[3] - rect[1]), edgecolor='y',
    #     #                           facecolor='none')
    #     # ax = plt.gca()
    #     # ax.add_patch(patch)
    #     # plt.show()
    #     # if ((frame) % 100 ==99 or frame == 0):
    #     #     plt.imshow(It1,cmap = 'gray')
    #     #     patch = patches.Rectangle((rect[0],rect[1]),(rect[2]-rect[0]),(rect[3]-rect[1]),edgecolor = 'y',facecolor='none')
    #     #     ax = plt.gca()
    #     #     ax.add_patch(patch)
    #     #     plt.show()
    np.save('sylvseqrects.npy', rects)

    sylvseqrects = np.load('sylvseqrects.npy')
    sylvseqrects_nobasis = np.load('sylvseqrects_nobasis-wcrt.npy')
    require = [1,200,300,350,400]
    for ind in range(len(require)):
        i = require[ind]
        fig = plt.figure()
        frame = sylvseq[:,:,i]
        rect_basis = sylvseqrects[i,:]
        # print(rect_basis)
        rect_nobasis = sylvseqrects_nobasis[i,:]
        # print(rect_nobasis)
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        patch1 = patches.Rectangle((rect_basis[0],rect_basis[1]),(rect_basis[2]-rect_basis[0]),(rect_basis[3]-rect_basis[1]),edgecolor = 'y',facecolor='none',linewidth=2)
        patch2 = patches.Rectangle((rect_nobasis[0],rect_nobasis[1]),(rect_nobasis[2]-rect_nobasis[0]),(rect_nobasis[3]-rect_nobasis[1]),edgecolor = 'g',facecolor='none',linewidth=2)

        ax = plt.gca()
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        # plt.show()
        fig.savefig('sylvseq_frame'+str(i)+'.png',bbox_inches='tight')
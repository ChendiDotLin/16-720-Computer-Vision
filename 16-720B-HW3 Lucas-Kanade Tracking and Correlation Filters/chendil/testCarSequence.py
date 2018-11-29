import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':

    carseq = np.load('../data/carseq.npy')
    # carseq = np.load('../data/sylvseq.npy')

    imH,imW,frames = np.shape(carseq)
    # print(imH,imW)
    print(frames)
    rect = [59.0,116.0,145.0,151.0]
    # rect = [101., 61., 155., 107.]

    rects = rect[:]
    for frame in range(frames-1):
        print(frame)
        It = carseq[:,:,frame]
        It1 = carseq[:,:,frame+1]
        p = LucasKanade.LucasKanade(It,It1,rect)
        rect[0] += p[0]
        rect[2] += p[0]
        rect[1] += p[1]
        rect[3] += p[1]
        # print(rect)
        rects = np.vstack((rects,rect))
        print
        # plt.imshow(It1, cmap='gray')
        # patch = patches.Rectangle((rect[0], rect[1]), (rect[2] - rect[0]), (rect[3] - rect[1]), edgecolor='y',
        #                           facecolor='none')
        # ax = plt.gca()
        # ax.add_patch(patch)
        # plt.show()
        if ((frame) % 100 == 99 or frame == 0):
            # since we are plotting rect on It1 (frame+1), frame 1, 100, 200, 300, 400 are plotted
            fig = plt.figure()
            plt.imshow(It1,cmap = 'gray')
            plt.axis('off')
            plt.axis('tight')
            patch = patches.Rectangle((rect[0],rect[1]),(rect[2]-rect[0]),(rect[3]-rect[1]),edgecolor = 'y',facecolor='none',linewidth=2)
            ax = plt.gca()
            ax.add_patch(patch)
            # plt.show()
            fig.savefig('carseq_frame'+str(frame+1)+'.png',bbox_inches='tight')
    np.save('carseqrects.npy', rects)
    # np.save('sylvseqrects_nobasis.npy', rects)
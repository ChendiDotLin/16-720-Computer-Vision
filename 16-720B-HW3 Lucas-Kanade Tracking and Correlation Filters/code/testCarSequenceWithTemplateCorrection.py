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
    rect0 = [59.0,116.0,145.0,151.0]
    # rect0 = [101., 61., 155., 107.]

    rect = rect0[:]
    rects = rect[:]
    threshold = 5
    update = True
    It = carseq[:,:,0]
    p0 = np.zeros(2)
    for frame in range(frames-1):
        print(frame)
        # It = carseq[:, :, frame]
        It1 = carseq[:,:,frame+1]
        p = LucasKanade.LucasKanade(It,It1,rect,p0)
        print(p)

        overall_p = p + [rect[0] - rect0[0],rect[1]-rect0[1]]
        pstar = LucasKanade.LucasKanade(carseq[:,:,0],It1,rect0,overall_p)
        # print(pstar)
        change = np.linalg.norm(overall_p-pstar)
        print(change)


        if change<threshold:
            p_here = (pstar - [rect[0] - rect0[0],rect[1]-rect0[1]])
            rect[0] += p_here[0]
            rect[2] += p_here[0]
            rect[1] += p_here[1]
            rect[3] += p_here[1]
            It = carseq[:, :, frame+1]
            rects = np.vstack((rects, rect))
            p0 = np.zeros(2)
        else:

            rects = np.vstack((rects,[rect[0]+p[0],rect[1]+p[1],rect[2]+p[0],rect[3]+p[1]]))
            p0 = p



            # print(rect)
        # plt.imshow(It1, cmap='gray')
        # patch = patches.Rectangle((rect[0], rect[1]), (rect[2] - rect[0]), (rect[3] - rect[1]), edgecolor='y',
        #                           facecolor='none')
        # ax = plt.gca()
        # ax.add_patch(patch)
        # plt.show()
        # if ((frame) % 100 == 99 or frame == 0):
        #     plt.imshow(It1,cmap = 'gray')
        #     patch = patches.Rectangle((rect[0],rect[1]),(rect[2]-rect[0]),(rect[3]-rect[1]),edgecolor = 'g',facecolor='none')
        #     ax = plt.gca()
        #     ax.add_patch(patch)
        #     plt.show()

    np.save('carseqrects-wcrt.npy', rects)
    # np.save('sylvseqrects_nobasis-wcrt.npy', rects)
    carseqrects = np.load('carseqrects.npy')
    carseqrects_ct = np.load('carseqrects-wcrt.npy')
    require = [1,100,200,300,400]
    for ind in range(len(require)):
        i = require[ind]
        fig = plt.figure()
        frame = carseq[:,:,i]
        rect_noct= carseqrects[i,:]
        rect_ct = carseqrects_ct[i,:]
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        patch1 = patches.Rectangle((rect_noct[0],rect_noct[1]),(rect_noct[2]-rect_noct[0]),(rect_noct[3]-rect_noct[1]),edgecolor = 'g',facecolor='none',linewidth=2)
        patch2 = patches.Rectangle((rect_ct[0],rect_ct[1]),(rect_ct[2]-rect_ct[0]),(rect_ct[3]-rect_ct[1]),edgecolor = 'y',facecolor='none',linewidth=2)

        ax = plt.gca()
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        # plt.show()
        fig.savefig('carseq-wcrt_frame' + str(i) + '.png',bbox_inches='tight')

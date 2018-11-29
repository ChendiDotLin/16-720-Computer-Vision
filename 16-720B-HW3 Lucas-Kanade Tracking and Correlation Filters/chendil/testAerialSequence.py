import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    aerialseq = np.load('../data/aerialseq.npy')
    # carseq = np.load('../data/sylvseq.npy')

    imH,imW,frames = np.shape(aerialseq)
    # print(imH,imW)
    print(frames)
    # rect = [101., 61., 155., 107.]

    for frame in range(frames-1):
        print(frame)
        image1 = aerialseq[:,:,frame]
        image2 = aerialseq[:,:,frame+1]
        mask = SubtractDominantMotion.SubtractDominantMotion(image1,image2)
        # print(mask.shape)
        # print(image2.shape)
        objects = np.where(mask == 1)
        # plt.imshow(image2, cmap='gray')
        # fig, = plt.plot(objects[1], objects[0], '*')
        # fig.set_markerfacecolor((0,0,1,0.8))
        # plt.show()
        # print(objects)
        if (frame == 29) or (frame == 59) or (frame == 89) or (frame ==119):
            # since we are plotting rect on image2 (frame+1), frame 30, 60, 90, 120 are plotted
            pic = plt.figure()
            plt.imshow(image2, cmap='gray')
            plt.axis('off')
            # print(mask)
            fig,= plt.plot(objects[1],objects[0] ,'*')
            fig.set_markerfacecolor((0, 0, 1, 1))
            # plt.show()
            pic.savefig('aerialseq_frame'+str(frame+1)+'.png', bbox_inches='tight')
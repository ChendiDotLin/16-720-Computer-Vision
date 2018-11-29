import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.imshow(1-bw,cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    all_boxes = []
    all_boxes.append([])
    line_num = 1
    bboxes.sort(key = lambda x:x[2])
    bottom = bboxes[0][2]
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if(minr >= bottom):
            bottom = maxr
            all_boxes.append([])
            line_num += 1
        all_boxes[line_num-1].append(bbox)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    for row in all_boxes:
        line = ""
        row.sort(key=lambda x: x[1])
        right = row[0][3]
        for box in row:
            minr, minc, maxr, maxc = box
            if (img == '01_list.jpg'):
                if (minc - right > 1.2 * (maxc - minc)):
                    line += " "
            else:
                if (minc - right > 0.8*(maxc-minc)):
                    line+= " "
            right = maxc
            letter = bw[minr:maxr, minc:maxc]
            #
            # print(letter)
            height,width = letter.shape
            # print(letter.shape)
            # letter = np.pad(letter,((width//5,height//5),(width//5,height//5)),'constant',constant_values=0.0)
            if (img == '01_list.jpg'):
                letter = np.pad(letter, ((20, 20), (20, 20)), 'constant', constant_values=0.0)
                letter = skimage.transform.resize(letter, (32, 32))
                letter = skimage.morphology.dilation(letter, skimage.morphology.square(1))
            else:
                letter = np.pad(letter,((50,50),(50,50)),'constant',constant_values=0.0)
                letter = skimage.transform.resize(letter,(32,32))
                letter = skimage.morphology.dilation(letter,skimage.morphology.square(2))
            letter = 1.0-letter
            #
            # plt.imshow(letter)
            # #
            # plt.show()

            letter = letter.T
            # print(letter)


            x = letter.reshape(1,32*32)
            h1 = forward(x, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            # print(probs.shape)
            idx = np.argmax(probs[0, :])
            # print(letters[idx])
            line+=(letters[idx])
        print(line)
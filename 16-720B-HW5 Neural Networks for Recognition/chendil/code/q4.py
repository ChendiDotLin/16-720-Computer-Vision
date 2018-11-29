import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    #
    image = skimage.filters.gaussian(image,sigma = 2,multichannel=True)
    image = skimage.color.rgb2grey(image)
    thresh = skimage.filters.threshold_otsu(image)
    bw = skimage.morphology.closing(image < thresh, skimage.morphology.square(10))
    cleared = skimage.segmentation.clear_border(bw)
    label_image,num = skimage.measure.label(cleared,background=0,return_num = True,connectivity = 2 )
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 200:
            bboxes.append(region.bbox)
    print(len(bboxes))
    return bboxes, bw
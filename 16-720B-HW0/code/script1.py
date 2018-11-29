from alignChannels import alignChannels
import numpy as np
import scipy.misc
import os

# Problem 1: Image Alignment


# 1. Load images (all 3 channels)
parent = os.path.dirname(os.getcwd())
data_path = parent+"/data"
red = np.load(data_path+'/red.npy')
green = np.load(data_path+'/green.npy')
blue = np.load(data_path+'/blue.npy')
print(red)
# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
dirName = parent+'/results'
if not os.path.exists(dirName):
    os.makedirs(dirName)
scipy.misc.imsave(dirName +'/rgb_output.jpg', rgbResult)

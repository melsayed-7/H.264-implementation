import matplotlib.pyplot as plt
from PIL import Image
from sys import getsizeof

import numpy as np
from helpers import *
from dwt import *



# converts rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



# reading # img_3 
img = plt.imread('t1.bmp')
img = rgb2gray(img)

# taking the first 400x400 pixels
img = img[:400,:400]


quantization_array = [4,19,31,49]

img = dwt(img, quantization_array)
img[0:img.shape[0]//2,0:img.shape[1]//2] = dwt(img[0:img.shape[0]//2,0:img.shape[1]//2], quantization_array)
img[0:img.shape[0]//4,0:img.shape[1]//4] = dwt(img[0:img.shape[0]//4,0:img.shape[1]//4], quantization_array)






img = idwt(img, quantization_array)


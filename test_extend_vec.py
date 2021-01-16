import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sys import getsizeof

import numpy as np
from helpers import *



def extend_vec(vec,extension_index):
    vec = list(vec)
    if(extension_index<0):
        ex = vec[1:-extension_index+1]
        ex.reverse()
        vec = ex +  vec

    elif(extension_index > 0):
        ex = vec[-extension_index-1:-1]
        ex.reverse()
        vec = vec + ex
    
    vec = np.asarray(vec)

    return vec
	

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


img = plt.imread('img_3.jpg')
img = rgb2gray(img)
img = img[:400,:400]


h0=np.array([-1/8 , 1/4 , 3/4 , 1/4 , -1/8])

h1=np.array([-1/2 , 1 , -1/2])

copy_img = np.zeros((img.shape[0], img.shape[1]+len(h0)-1))
r= np.zeros((img.shape[0], img.shape[1]+len(h0)-1))

for i in range(copy_img.shape[0]):
	vec = extend_vec(img[i,:],-(len(h0)-1)//2)
	vec = extend_vec(vec,(len(h0)-1)//2)
    r[i,:] = vec

copy_img[i,:] = vec
print(copy_img[i,:])

x= 1
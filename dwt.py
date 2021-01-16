import matplotlib.pyplot as plt
from PIL import Image
from sys import getsizeof

import numpy as np
from helpers import *



# converts rgb to gray
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



# reading # img_3 
img = plt.imread('img_3.jpg')
img = rgb2gray(img)

# taking the first 400x400 pixels
img = img[:400,:400]

# img = np.ones((64,64))*127



# plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


# takes a vector and a filter and do normal convolution
def filter(vec,f):
	newvec = np.convolve(vec, f)
	return newvec 


# down samples the vector to half
def downsample(vec):
	vec=vec[::2]
	return vec



# upsamples by interpolating 0's
def upsample(vec):
    length = len(vec)
    newvec = np.zeros((2*length))
    for i in range(length):
        newvec[2*i]=vec[i]  
        
    # for i in range(1,len(newvec)-1,2):
    #     newvec[i]=(newvec[i-1]+newvec[i+1])/2 
    return newvec



import numpy as np



# takes a vector and a value positive if append at the end and negative if at the beginning
# and then flips the abs(value) pixels to pad the vector
def extend_vec(vec,extension_value):
    vec = list(vec)
    if(extension_value<0):
        ex = vec[1:-extension_value+1]
        ex.reverse()
        vec = ex +  vec

    elif(extension_value > 0):
        ex = vec[-extension_value-1:-1]
        ex.reverse()
        vec = vec + ex

    return np.asarray(vec)

	
x = [1,1,1,1,1]
y = [1,1,1,1,1]

ll = []



############################################
#discrete wavelet transform


# takes an image and an array n that represents the quantization values

def dwt(img,n):


	#taking a copy of the img
	copy_img = np.copy(img)


	h0=np.array([-1/8 , 1/4 , 3/4 , 1/4 , -1/8])

	h1=np.array([-1/2 , 1 , -1/2])

	rows, columns = img.shape


	

	######################
	# here I do filtering either low or high on the x axis then on the y axis
	# with the extension for correct convolution and then downsample
	# I repeat this for 4 times
	# then assign the final output in ll_2, lh_2, hl_2, and hh_2  ( those are the 4 block each assigned in its place)

	#     ll   lh
	#     hl   hh   

	# copy img and copy_img_2 are  just helper matrcies to easily downsample and operate on the data
	 
	copy_img = np.zeros((img.shape[0], img.shape[1]+len(h0)-1))
	for i in range(copy_img.shape[0]):
		vec = extend_vec(img[i,:],-(len(h0)-1)//2)
		vec = extend_vec(vec,(len(h0)-1)//2)
		copy_img[i,:] = vec

	copy_img_2 = np.zeros((copy_img.shape[0]+len(h0)-1, copy_img.shape[1]))
	for i in range(copy_img_2.shape[1]):
		vec = extend_vec(copy_img[:,i],-(len(h0)-1)//2)
		vec = extend_vec(vec,(len(h0)-1)//2)
		copy_img_2[:,i] = vec

	ll_1 = np.zeros((copy_img_2.shape[0], img.shape[1]//2))
	ll_2 = np.zeros((img.shape[0]//2, ll_1.shape[1]))
	


	for row in range(rows):
		ll_1[row,:] = downsample(filter(copy_img_2[row,:], h0)[len(h0)-1:-len(h0)+1])


	
	for column in range(ll_1.shape[1]):
		ll_2 [:,column] = downsample(filter(ll_1[:, column], h0)[len(h0)-1:-len(h0)+1])
	


	img[0:img.shape[0]//2,0:img.shape[1]//2] = ll_2//n[0]
	



	#######################

	copy_img = np.zeros((img.shape[0], img.shape[1]+len(h0)-1))
	for i in range(copy_img.shape[0]):
		vec = extend_vec(img[i,:],-(len(h0)-1)//2)
		vec = extend_vec(vec,(len(h0)-1)//2)
		copy_img[i,:] = vec

	copy_img_2 = np.zeros((copy_img.shape[0]+len(h1)-1, copy_img.shape[1]))
	for i in range(copy_img_2.shape[1]):
		vec = extend_vec(copy_img[:,i],-(len(h1)-1))
		copy_img_2[:,i] = vec

	lh_1 = np.zeros((copy_img_2.shape[0], img.shape[1]//2))
	lh_2 = np.zeros((img.shape[0]//2, ll_1.shape[1]))
	
	
	for row in range(rows):
		lh_1[row,:] = downsample(filter(copy_img_2[row,:], h0)[len(h0)-1:-len(h0)+1])
	
	for column in range(lh_1.shape[1]):
		lh_2 [:,column] = downsample(filter(lh_1[:,column], h1)[len(h1)-1:-len(h1)+1])


	img[0:img.shape[0]//2,img.shape[1]//2:] = lh_2//n[1]

	#######################

	copy_img = np.zeros((img.shape[0], img.shape[1]+len(h1)-1))
	for i in range(copy_img.shape[0]):
		vec = extend_vec(img[i,:],-(len(h1)-1))
		copy_img[i,:] = vec

	copy_img_2 = np.zeros((copy_img.shape[0]+len(h0)-1, copy_img.shape[1]))
	for i in range(copy_img_2.shape[1]):
		vec = extend_vec(copy_img[:,i],-(len(h0)-1)//2)
		vec = extend_vec(vec,(len(h0)-1)//2)
		copy_img_2[:,i] = vec

	hl_1 = np.zeros((copy_img_2.shape[0], img.shape[1]//2))
	hl_2 = np.zeros((img.shape[0]//2, hl_1.shape[1]))



	for row in range(rows):
		hl_1[row,:] = downsample(filter(copy_img_2[row,:], h1)[len(h1)-1:-len(h1)+1])
	
	for column in range(hl_1.shape[1]):
		hl_2 [:,column] = downsample(filter(hl_1[:,column], h0)[len(h0)-1:-len(h0)+1])


	img[img.shape[0]//2:,0:img.shape[1]//2] = hl_2//n[2]


	#######################
	copy_img = np.zeros((img.shape[0], img.shape[1]+len(h1)-1))
	for i in range(copy_img.shape[0]):
		vec = extend_vec(img[i,:],-(len(h1)-1))
		copy_img[i,:] = vec

	copy_img_2 = np.zeros((copy_img.shape[0]+len(h1)-1, copy_img.shape[1]))
	for i in range(copy_img_2.shape[1]):
		vec = extend_vec(copy_img[:,i],-(len(h1)-1))
		copy_img_2[:,i] = vec

	hh_1 = np.zeros((copy_img_2.shape[0], img.shape[1]//2))
	hh_2 = np.zeros((img.shape[0]//2, hh_1.shape[1]))
	
	for row in range(rows):
		hh_1[row,:] = downsample(filter(copy_img[row,:], h1)[len(h1)-1:-len(h1)+1])
	
	for column in range(hh_1.shape[1]):
		hh_2 [:,column] = downsample(filter(hh_1[:,column], h1)[len(h1)-1:-len(h1)+1])


	img[img.shape[0]//2:,img.shape[1]//2:] = hh_2//n[3]

	#######################


	return img





quantization_array = [4,19,31,49]

img = dwt(img, quantization_array)
img[0:img.shape[0]//2,0:img.shape[1]//2] = dwt(img[0:img.shape[0]//2,0:img.shape[1]//2], quantization_array)
img[0:img.shape[0]//4,0:img.shape[1]//4] = dwt(img[0:img.shape[0]//4,0:img.shape[1]//4], quantization_array)




###########################################################


# 2d to 1d
one_d = takestwoD(img)

# run length
run_length_encoded = run_length(one_d)


# huffman encoding 
huffman  = Huffman_encoding()
huffman_encoded = huffman.compress(run_length_encoded)


from sys import getsizeof

# getting the size of raw image
print("size of image in bits:")
print(getsizeof(one_d))
print("\n\n")

# print(len(run_length_encoded))




#####
# doing decoding

# size of compressed image
print("size of compressed in bits:")
print(len(huffman_encoded))
print("\n\n")
huffman_decoded = huffman.decode_text(huffman_encoded)
run_length_decoded = reverse_run_length(run_length_encoded)


two_d = takesoneD(run_length_decoded, img.shape[0], img.shape[1])





############################################################################
# inverse discrete wavelet transform


# here is the inverse dwt

# almost the same operations but reversed

def idwt(img,n):


	g0=np.array([1/2 , 1 , 1/2])  #-2  -1  0

	g1=np.array([ -1/8 , -1/4 , 3/4 , -1/4 , -1/8 ])  #  -1   0  1  2  3

	rows , columns = img.shape		


	#creating 2 helper arrays to manipulate the input 
	# then do the operation 4 times

	# x_1 is what is retreived from ll block
	# x_2 is what is retreived from lh block
	# x_3 is what is retreived from hl block
	# x_4 is what is retreived from hh block



	x1 = np.zeros((img.shape[0],img.shape[1]//2))
	x3 = np.zeros((img.shape[0],img.shape[1]//2))



	for column in range(columns//2):
		upsampled = upsample(img[0:img.shape[0]//2,column]*n[0])
		upsampled = extend_vec(upsampled,-(len(g0)-1))
		
	for column in range(columns//2):
		upsampled = upsample(img[img.shape[0]//2:,column]*n[2])
		upsampled = extend_vec(upsampled,-(len(g0)-1))
		x1[:,column] = filter(upsampled, g0)[len(g0)-1:-len(g0)+1]


	x2 = np.zeros((img.shape[0],img.shape[1]//2))
	x4 = np.zeros((img.shape[0],img.shape[1]//2))

	for column in range(columns//2):
		upsampled = upsample(img[img.shape[0]//2:,column]*n[1])
		upsampled = extend_vec(upsampled,-1)
		upsampled = extend_vec(upsampled, 3)
		x2[:,column] = filter(upsampled, g1)[len(g1)-1:-len(g1)+1]


	for column in range(columns//2):
		upsampled = upsample(img[img.shape[0]//2:,column]*n[1])
		upsampled = extend_vec(upsampled,-1)
		upsampled = extend_vec(upsampled, 3)
		x4[:,column] = filter(upsampled, g1)[len(g1)-1:-len(g1)+1]

	
	######################


	x_1 = x1 + x2
	x_2 = x3 + x4

	x_11 = np.zeros(img.shape)
	x_22 = np.zeros(img.shape)


	for row in range(x_1.shape[0]):
		upsampled = upsample(x_1[row,:])
		upsampled = extend_vec(upsampled,-(len(g0)-1))
		x_11[row,:] = filter(upsampled, g0)[len(g0)-1:-len(g0)+1]


	for column in range(columns//2):
		upsampled = upsample(x_1[row,:])
		upsampled = extend_vec(upsampled,-1)
		upsampled = extend_vec(upsampled, 3)
		x_22[:,column] = filter(upsampled, g1)[len(g1)-1:-len(g1)+1]


	return (x_11 + x_22)






# img[0:img.shape[0]//4,0:img.shape[1]//4] = idwt(img[0:img.shape[0]//4,0:img.shape[1]//4],[1,1,1,1])
# img[0:img.shape[0]//2,0:img.shape[1]//2] = idwt(img[0:img.shape[0]//2,0:img.shape[1]//2], [1,1,1,1])
# img = idwt(img,[1,1,1,1])

# img = two_d

img[0:img.shape[0]//4,0:img.shape[1]//4] = idwt(img[0:img.shape[0]//4,0:img.shape[1]//4], quantization_array)
img[0:img.shape[0]//2,0:img.shape[1]//2] = idwt(img[0:img.shape[0]//2,0:img.shape[1]//2], quantization_array)
img = idwt(img, quantization_array)


print(img.shape)
plt.imshow(img, cmap = "gray")
#print(img)





import numpy as np
import scipy.ndimage
import os,time
import skimage.io
import deep_recog

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	#x = x.astype('float')
	#image = deep_recog.preprocess_image(x)

	x = skimage.transform.resize(x, (224,224))
	h,w,channel = np.shape(x)
	if channel == 1: # grey
		x = np.matlib.repmat(x,1,1,3)
	if channel == 4: # special case
		x = x[:,:,0:3]
	mean = [0.485,0.456,0.406]
	std = [0.229,0.224,0.225]
	for i in range(3):
		x[:,:,i] = (x[:,:,i] - mean[i]) / std[i]
	L = len(vgg16_weights)
	linear_count = 0
	for layer in range(L):
		weights = vgg16_weights[layer]
		#print(weights[0])
		if (weights[0] == 'conv2d'):
			x = multichannel_conv2d(x,weights[1],weights[2])
		elif (weights[0] == 'linear'):
			x = linear(x,weights[1],weights[2])
			linear_count+=1
			if(linear_count ==2):
				break #stop at fc7
		elif (weights[0] == 'maxpool2d'):
			x = max_pool2d(x,weights[1])
		elif (weights[0] == 'relu'):
			x = relu(x)

	#xcompare = vgg16(image)

	#print(np.linalg.norm(x-xcompare.detach().numpy()))
	#feat = relu(x)
	#print(np.shape(x)[0])
	feat = x.reshape(1,np.shape(x)[0])
	return feat


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''

	h,w,input_dim  = np.shape(x)
	output_dim,input_dim,kernel_size,kernel_size = np.shape(weight)
	feat = np.array([], dtype=np.int64).reshape(h,w,0)
	#feat = []
	for j in range(output_dim):
		sum = np.zeros((h,w,1))
		for k in range(input_dim):
			signal = x[:,:,k]
			filter = weight[j,k,:]
			filter = np.flipud(np.fliplr(filter))
			sum[:,:,0] += scipy.ndimage.convolve(signal,filter,mode = 'constant')
		sum += bias[j]
		feat = np.dstack((feat,sum))

		#feat.append(sum)
	#print(np.shape(feat))
	return (feat)

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(0,x)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''

	H,W,input_dim = np.shape(x)


	Hsize = int(H/ size)
	Wsize = int(W/size)
	return(x.reshape(Hsize, size, Wsize, size,input_dim).max(axis=(1, 3)))


def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	# using tranpose to match the output of pytorch
	y = np.matmul(W,np.transpose(x).flatten()) + b
	return (y)


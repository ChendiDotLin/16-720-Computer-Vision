import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	[m,n,channel] = np.shape(image)

	# make sure that entries in image are float and with range 0 1
	if (type(image[0,0,0]) == int ):
		image = image.astype('float') / 255
	elif (np.amax(image) > 1.0):
		image = image.astype('float') / 255

	if channel == 1: # grey
		image = np.matlib.repmat(image,1,1,3)
	if channel == 4: # special case
		image = image[:,:,0:3]
	channel = 3
	image = skimage.color.rgb2lab(image)
	scale = [1,2,4,8,8 * np.sqrt(2)]
	F = len(scale) * 4
	response = np.zeros((m, n, 3*F))
	#for i in range(channel):
	#	for j in range (len(scale)):
	#		response[:,:,i*len(scale)*4+j*4] = scipy.ndimage.gaussian_filter(image[:,:,i],sigma = scale[j],output=np.float64) # guassian
	#		response[:,:,i*len(scale)*4+j*4+1] = scipy.ndimage.gaussian_laplace(image[:,:,i],sigma = scale[j],output=np.float64) # guassian laplace
	#		response[:,:,i * len(scale)*4 + j*4+2] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order = [0,1],output = np.float64) # derivative in x direction
	#		response[:, :, i * len(scale)*4 + j*4+3] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order=[1, 0],output = np.float64)  # derivative in y direction


	# ----- TODO -----
	for i in range(channel):
		for j in range (len(scale)):
			response[:,:,channel*4*j+i] = scipy.ndimage.gaussian_filter(image[:,:,i],sigma = scale[j],output=np.float64) # guassian
			response[:,:,channel*4*j+3+i] = scipy.ndimage.gaussian_laplace(image[:,:,i],sigma = scale[j],output=np.float64) # guassian laplace
			response[:,:,channel*4*j+6+i] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order = [0,1],output = np.float64) # derivative in x direction
			response[:,:,channel*4*j+9+i] = scipy.ndimage.gaussian_filter(image[:,:,i], sigma = scale[j], order=[1, 0],output = np.float64)  # derivative in y direction
	return response






def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----

	response = extract_filter_responses(image)
	m,n,filnum = np.shape(response)
	k,filnum = np.shape(dictionary)
	dis = np.zeros(k)
	wordmap = np.zeros((m,n))
	for i in range(m):
		for j in range(n):
			pixel = response[i][j][:]
			pixel = np.reshape(pixel,(1,filnum))
			#print(np.shape(pixel))
			# for kk in range(k):
			# 	word = dictionary[kk]
			# 	dis[kk] = scipy.spatial.distance.cdist(pixel,word)
			dis = scipy.spatial.distance.cdist(dictionary,pixel)
			#print(np.shape(dis))
			#print(np.unravel_index(np.argmax(dis,axis = None),dis.shape)[0])
			wordmap[i,j] = np.unravel_index(np.argmin(dis,axis = None),dis.shape)[0]
	# plt.imshow(wordmap,cmap = 'rainbow')
	# plt.show()
	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''


	i,alpha,image_path = args
	#print("../data/" + image_path[i][0])
	image = skimage.io.imread("../data/" + image_path[i][0])
	#image = image.astype('float') / 255
	filter_responses = extract_filter_responses(image)
	# ----- TODO -----


	m,n,kk = np.shape(filter_responses)
	sampled_response = np.reshape(filter_responses,(m*n,kk))
	idx = np.random.randint(m*n, size= alpha)
	sampled_response = sampled_response[idx,:]
	# pick up alpha random pixels
	np.save('sampled_response.npy',sampled_response)
	return sampled_response





def compute_dictionary(num_workers = 2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)T = 200T = f
	'''



	train_data = np.load("../data/train_data.npz")
	#print(np.shape(train_data['image_names']))
	#print(train_data['image_names'])
	# ----- TODO -----
	T = np.shape(train_data['image_names'])[0]
	#T = 200
	alpha = 250
	K = 200
	#filter_responses = np.zeros((alpha*T,3*20))
	filter_responses = np.array([], dtype=np.int64).reshape(0,3*20)
	for i in range (int(T/num_workers)):
		p = multiprocessing.Pool(num_workers)
		param = []
		for j in range(num_workers):
			param.append((i*num_workers+j,alpha,train_data['image_names']))
		[fil1,fil2,fil3,fil4] = p.map(compute_dictionary_one_image,param)
		# somehow concate them
		#fil = np.concatenate((fil1,fil2,fil3,fil4),axis = 0)
		#print(np.shape(fil))
		filter_responses = np.vstack([filter_responses,fil1,fil2,fil3,fil4])
		#filter_responses = np.concatenate((filter_responses,fil),axis = 0)
		#filter_responses[i*num_workers*alpha:(i+1)*num_workers*alpha,:] = fil
		#filter_responses[i*alpha:(i+1)*alpha,:] = compute_dictionary_one_image((i,alpha,train_data['image_names']))
	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs = -1).fit(filter_responses)
	dictionary = kmeans.cluster_centers_
	np.save('dictionary.npy',dictionary)

	return dictionary



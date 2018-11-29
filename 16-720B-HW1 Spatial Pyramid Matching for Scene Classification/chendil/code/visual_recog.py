import numpy as np
import threading
import queue
import skimage.color
import imageio
import os,time
import math
import visual_words
import matplotlib.pyplot as plt


def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''



	train_data = np.load("../data/train_data.npz")
	dictionary = np.load("dictionary.npy")
	# ----- TODO -----
	N = np.shape(train_data['image_names'])[0]
	layernum = 3
	K,filnum = np.shape(dictionary)
	M = int(K*(4**layernum-1)/3)
	features = np.array([], dtype=np.int64).reshape(0,M)
	labels = []
	for i in range(N):
		filepath =  train_data['image_names'][i][0]
		folder = filepath.split('/')[0]
		#print(folder)
		labels.append(compute_label(folder))

		feature = get_image_feature(filepath,dictionary,layernum,K)
		features = np.vstack([features,feature])
	#print(np.shape(features))
	#print(np.shape(labels))
	np.save('trained_system.npy', features)
	np.save('labels.npy', labels)
	np.savez('trained_system.npz',features = features,labels = labels,dictionary = dictionary,layernum = layernum)

	pass

def compute_label(folder):
	if folder == 'auditorium':
		return(0)
	elif folder == 'baseball_field':
		return(1)
	elif folder == 'desert':
		return(2)
	elif folder == 'highway':
		return(3)
	elif folder == 'kitchen':
		return(4)
	elif folder == 'laundromat':
		return(5)
	elif folder == 'waterfall':
		return(6)
	else:
		return(7)

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")
	# ----- TODO -----
	trained_features = trained_system['features']
	trained_labels = trained_system['labels']
	trained_dictionary = trained_system['dictionary']
	trained_SPM_layer_num = int(trained_system['layernum'])
	N_test = np.shape(test_data['image_names'])[0]
	K,filnum = np.shape(trained_dictionary)
	confusion = np.zeros((8,8))
	for i in range(N_test):
		print(i) # track progress
		filepath =  test_data['image_names'][i][0]
		folder = filepath.split('/')[0]
		#print(filepath)
		test_label = (compute_label(folder))
		#print(test_label)
		test_feature = get_image_feature(filepath,trained_dictionary,trained_SPM_layer_num,K)
		similarity = distance_to_set(test_feature,trained_features)
		idx = np.unravel_index(np.argmax(similarity, axis=None), similarity.shape)[0]
		predict_label = trained_labels[idx]
		confusion[test_label,predict_label] += 1
		accuracy = np.trace(confusion)/N_test
	return(confusion,accuracy)




def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K*(4^layer_num-1)/3))
	'''



	# ----- TODO -----
	image = skimage.io.imread("../data/" + file_path)
	wordmap = visual_words.get_visual_words(image,dictionary)
	feature = get_feature_from_wordmap_SPM(wordmap,layer_num,K)
	return feature

def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	#pass



	# ----- TODO -----

	intersection = np.minimum(histograms,word_hist)
	similarity = np.sum(intersection,axis = 1)
	return similarity

def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	# ----- TODO -----
	h,w = np.shape(wordmap)
	#print (h,w)
	#print(wordmap)
	data = np.reshape(wordmap,(1,h*w))
	bin = np.linspace(0,dict_size,num = dict_size+1,endpoint = True)
	hist, bin_edges = np.histogram(data, bins= bin, density=True)
	#print(hist)
	# plt.bar(bin_edges[:-1], hist, width=1)
	# plt.xlim(min(bin_edges), max(bin_edges))
	# plt.show()
	hist = np.reshape(hist,(1,dict_size))
	#print(np.shape(hist))
	#print(np.sum(hist))
	return (hist)



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	h,w = np.shape(wordmap)
	weights = []
	result = np.array([], dtype=np.int64).reshape(1,0)
	#print (type(layer_num))
	for l in range(layer_num):
		if l == 0 or l == 1:
			weights.append(2**(-(layer_num-1)))
		else:
			weights.append(2**(l-layer_num))
	for i in range(len(weights)): # do it from top to bottom
		layer = len(weights)-1-i
		weight = weights[len(weights)-1-i]
		subh = int(h/(2**layer))
		subw = int(w/(2**layer))
		#print (h/(2**layer))
		#print (w/(2**layer))
		#if i==0:#finest layer
		for row in range(2**layer):
			for col in range(2**layer):
				subword = wordmap[subh*row:subh*(row+1),subw*col:subw*(col+1)]
				hist = get_feature_from_wordmap(subword,dict_size)

				result = np.hstack([result,hist*weight])
				#print(np.shape(result))
	#print(np.shape(result))
	#print (np.sum(result))
	return(result)






	


import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	train_data = np.load("../data/train_data.npz")
	# ----- TODO -----
	N = np.shape(train_data['image_names'])[0]
	features = np.array([], dtype=np.int64).reshape(0,4096)
	labels = []
	for i in range(N):
		print(i) # track progress
		filepath =  train_data['image_names'][i][0]
		folder = filepath.split('/')[0]
		#print(folder)
		labels.append(compute_label(folder))
		feature = get_image_feature((i,filepath,vgg16))
		features = np.vstack([features,feature])
	#print(np.shape(label))
	np.save('deep_trained_system.npy', features)
	#labels = np.load('labels.npy')
	np.save('labels.npy', labels)
	np.savez('deep_trained_system.npz',features = features,labels = labels)
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

def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	
	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("deep_trained_system.npz")

	# ----- TODO -----
	trained_features = trained_system['features']
	trained_labels = trained_system['labels']
	N_test = np.shape(test_data['image_names'])[0]
	confusion = np.zeros((8,8))
	#vgg16.weights = util.get_VGG16_weights()

	for i in range(	N_test):
		print(i) # track progress
		filepath =  test_data['image_names'][i][0]
		folder = filepath.split('/')[0]
		#print(folder)
		test_label = (compute_label(folder))
		#image =  skimage.io.imread("../data/"+ filepath)

		#test_feature = network_layers.extract_deep_feature(image,vgg16.weights)
		test_feature = get_image_feature((i,filepath,vgg16))
		similarity = distance_to_set(test_feature,trained_features)
		idx = np.unravel_index(np.argmax(similarity, axis=None), similarity.shape)[1]
		#print(idx)
		predict_label = trained_labels[idx]
		confusion[test_label,predict_label] += 1
	accuracy = np.trace(confusion)/N_test
	print(accuracy)
	return(confusion,accuracy)



def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	# ----- TODO -----
	#image = image.astype('float')
	image = skimage.transform.resize(image, (224,224))
	h,w,channel = np.shape(image)
	if channel == 1: # grey
		image = np.matlib.repmat(image,1,1,3)
	if channel == 4: # special case
		image = image[:,:,0:3]
	#print(image)
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
											   torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
	image = transform(image)
	image = image.unsqueeze(0)
	return(image)

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i,image_path,vgg16 = args
	#print("../data/" + image_path[i][0])
	#image = skimage.io.imread("../data/" + image_path[i][0])
	image = skimage.io.imread("../data/"+ image_path)
	image = preprocess_image(image)
	# ----- TODO -----
	#feat = vgg16(image)
	top_layers = torch.nn.Sequential(*list(vgg16.children())[0])
	fc7 = torch.nn.Sequential(*list(vgg16.children())[1][:4])

	feat = fc7(top_layers(image).flatten())
	feat = feat.detach().numpy().reshape(1,4096)
	np.save('feat.npy',feat)
	return (feat)




def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	# ----- TODO -----
	
	dist = scipy.spatial.distance_matrix(feature,train_features,p=2)
	return (-1*dist)
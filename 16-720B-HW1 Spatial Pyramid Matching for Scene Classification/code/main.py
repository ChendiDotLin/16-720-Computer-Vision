import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io
import network_layers

if __name__ == '__main__':

	num_cores = util.get_num_CPU()
	path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"

# Q1
	image = skimage.io.imread(path_img)
# 	image = image.astype('float') / 255
# 	image = skimage.transform.resize(image, (224,224))
	filter_responses = visual_words.extract_filter_responses(image)
	#sampled_response = visual_words.compute_dictionary_one_image((1,100,path_img))
	#print(np.shape(sampled_response))

	util.display_filter_responses(filter_responses)
	num_cores = 4
	#dictionary = visual_words.compute_dictionary(num_workers=num_cores)
	#print(np.shape(dictionary))
	# dictionary = np.load('dictionary.npy')
	#img = visual_words.get_visual_words(image,dictionary)
	#plt.imshow(img,cmap = 'rainbow')
	#plt.show()
	#util.save_wordmap(wordmap, filename)
	#k,filnum = np.shape(dictionary)
	#visual_recog.get_feature_from_wordmap(img,k)

	# Q2
	# visual_recog.build_recognition_system(num_workers=num_cores)
	# layernum = 3
	# features = np.load('features.npy')
	# labels = np.load('labels.npy')
	# np.savez('trained_system.npz', features=features, labels=labels, dictionary=dictionary, layernum=layernum)
	#print(np.load('trained_system.npz').files)
	#conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	#print(conf)
	# print (accuracy)
	#np.savetxt("conf.csv", conf.astype(int),fmt = '%i', delimiter=",")

	#print(np.diag(conf).sum()/conf.sum())


# Q3 deep learning
	vgg16 = torchvision.models.vgg16(pretrained=True).double()
	vgg16.eval()
	vgg16_weights = util.get_VGG16_weights()
	#print(type(vgg16_weights))
	#print (np.shape(network_layers.extract_deep_feature(image,vgg16_weights))) # a test of network_layers
	#result = deep_recog.preprocess_image(image)
	#print (result)
	#print (np.shape(result))
	#result = deep_recog.get_image_feature((0,path_img,vgg16))
	#print(np.shape(result))
	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	conf,accuracy = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	print(conf)
	print(np.diag(conf).sum()/conf.sum())
	#np.savetxt("deep_conf.csv", conf.astype(int),fmt = '%i', delimiter=",")

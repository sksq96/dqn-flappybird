import os
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle

test_folder = 'notMNIST_small'
train_folder = 'notMNIST_large'

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder):
	print(folder)

	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size))
	
	image_idx = 0 # idx and image_idx will be different in case of exception raised
	for idx,image in enumerate(os.listdir(folder)):
		
		image_file = os.path.join(folder, image)
		
		try:
			image_data = (ndimage.imread(image_file) - pixel_depth//2) // pixel_depth
			
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))

			dataset[image_idx] = image_data

			if image_idx >= 5000:
				break

			image_idx += 1
		
		except:
			print('Error file:', idx+1)
	
	num_images = image_idx
	dataset = dataset[0:num_images, :, :]
	
	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset), end='\n\n')
	
	return dataset

def load(folder):
	for letter_folder in os.listdir(folder):
		dataset = load_letter(os.path.join(folder, letter_folder))

		pickle_filename = letter_folder + '.pk'
		with open(pickle_filename, 'wb') as f:
			pickle.dump(dataset, f)

load(test_folder)
load(train_folder)


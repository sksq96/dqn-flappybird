import os
import numpy as np
from six.moves import cPickle as pickle

train_folder = 'pickle/train'
test_folder = 'pickle/test'

image_size = 28  # Pixel width and height.
np.random.seed(133)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    
    return shuffled_dataset, shuffled_labels


def pickle_todata(folder):
    dataset = np.ndarray(shape=(0, image_size*image_size))
    labels = np.ndarray(shape=(0))

    for pickle_file in os.listdir(folder):
        print(pickle_file)

        label, _ = pickle_file.split('.')
        pickle_file_path = os.path.join(folder, pickle_file)

        with open(pickle_file_path, 'rb') as f:
            label_data = pickle.load(f)

        n_samples = len(label_data)
        
        label_data = label_data.reshape((n_samples, -1))[:]
        dataset = np.append(dataset, label_data, axis=0)[:]
        
        labels = np.append(labels, [label]*n_samples)[:]

    return dataset, labels

Xr, yr = pickle_todata(train_folder)
# Xr, yr = randomize(Xr, yr)

Xe, ye = pickle_todata(test_folder)
# Xe, ye = randomize(Xe, ye)

save = {
    'Xr': Xr,
    'yr': yr,
    'Xe': Xe,
    'ye': ye
}

with open('train_test.pk', 'wb') as f:
    pickle.dump(save, f)


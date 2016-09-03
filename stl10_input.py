import os, sys, tarfile, urllib
import numpy as np
import tensorflow as tf

# image shape
H = 96
W = 96
C = 3

DATA_DIR = '/scratch/clear/abietti/data/stl10_binary'
WHITENED_CKN_MODEL = '/scratch/clear/abietti/results/ckn/stl10_white_py/layers_1.npy'


def params():
    return {
        'n_classes': 10,
        'lmbda': 4e-4,
        'lrs': [0.05, 0.1, 0.5],
        'results_root': '/scratch/clear/abietti/results/ckn/stl10_white_py/accs',
        'ckn_batch_size': 32,
        'encode_size': 2000,
    }


def load_ckn_layers_whitened():
    return np.load(WHITENED_CKN_MODEL)


def load_train_val_white(fold=0):
    X = np.load(os.path.join(DATA_DIR, 'train_X_white.npy'))
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin')).astype(np.int32)

    idxs = fold_idxs(fold)
    idxs_val = np.setdiff1d(np.arange(X.shape[0]), idxs)

    return X[idxs], y[idxs], X[idxs_val], y[idxs_val]


def load_train_test_white(fold=0):
    X = np.load(os.path.join(DATA_DIR, 'train_X_white.npy'))
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin'))
    Xt = np.load(os.path.join(DATA_DIR, 'test_X_white.npy'))
    yt = read_labels(os.path.join(DATA_DIR, 'test_y.bin'))

    idxs = fold_idxs(fold)

    return X[idxs], y[idxs].astype(np.int32), Xt, yt.astype(np.int32)


def load_train_test_full_white():
    X = np.load(os.path.join(DATA_DIR, 'train_X_white.npy'))
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin'))
    Xt = np.load(os.path.join(DATA_DIR, 'test_X_white.npy'))
    yt = read_labels(os.path.join(DATA_DIR, 'test_y.bin'))

    return X, y.astype(np.int32), Xt, yt.astype(np.int32)


def fold_idxs(fold):
    folds = open(os.path.join(DATA_DIR, 'fold_indices.txt')).readlines()
    return np.fromstring(folds[fold], dtype=int, sep=' ')


def augmentation(image):
    # pad with zeros to make the image 100x100
    image = tf.image.resize_image_with_crop_or_pad(image, 100, 100)

    # extract random 96x96 crops
    image = tf.random_crop(image, size=[96, 96, 3])

    return image


#### Functions from https://github.com/mttk/STL10 ####

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels - 1  # change range from [1..10] to [0..9]


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

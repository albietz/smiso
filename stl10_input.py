import os, sys, tarfile, urllib
import numpy as np
import tensorflow as tf

# image shape
H = 96
W = 96
C = 3

DATA_DIR = '/scratch/clear/abietti/data/stl10_binary'
MODEL_DIR = '/scratch/clear/abietti/results/ckn/stl10_white_3_11'


def params():
    return {
        'n_classes': 10,
        # 'lmbda': [0.01, 1e-4, 1e-6, 1e-8],
        'lmbda': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'lrs': [0.1, 1.0],
        'miso_lrs': [0.1, 1.0],
        'saga_lrs': [0.1, 1.0],
        'results_root': os.path.join(MODEL_DIR, 'accs'),
        'ckn_batch_size': 64,
        'encode_size': 4000,
    }


def params_scat():
    return {
        'n_classes': 10,
        # 'lmbda': [1e-3, 1e-4, 1e-5],
        'lmbda': [1e-5],
        'lrs': [0.1, 1.0],
        'prox': 'l1',
        'prox_weight': 1e-4,
        'miso_lrs': [0.1, 1.0],
        'saga_lrs': [0.1, 1.0],
        'results_root': os.path.join(MODEL_DIR, 'accs_scat'),
        'encode_size': 2000,
    }


def load_ckn_layers_whitened():
    return np.load(os.path.join(MODEL_DIR, 'layers_1.npy'))


def get_scattering_params():
    from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
    filters, lw = multiresolution_filter_bank_morlet2d(64, J=4, L=4, sigma_phi=0.8, sigma_xi=0.8)
    m = 2
    return filters, m


def load_train_val_white(fold=0):
    X = np.load(os.path.join(DATA_DIR, 'train_X_white.npy'))
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin')).astype(np.int32)

    idxs = fold_idxs(fold)
    idxs_val = np.setdiff1d(np.arange(X.shape[0]), idxs)

    return X[idxs], y[idxs], X[idxs_val], y[idxs_val]


def load_train_val_raw(fold=0):
    X = read_all_images(os.path.join(DATA_DIR, 'train_X.bin')).astype(np.float32) / 255
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


def load_train_test_raw(fold=0):
    X = read_all_images(os.path.join(DATA_DIR, 'train_X.bin')).astype(np.float32) / 255
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin'))
    Xt = read_all_images(os.path.join(DATA_DIR, 'test_X.bin')).astype(np.float32) / 255
    yt = read_labels(os.path.join(DATA_DIR, 'test_y.bin'))

    idxs = fold_idxs(fold)

    return X[idxs], y[idxs].astype(np.int32), Xt, yt.astype(np.int32)


def load_train_test_full_white():
    X = np.load(os.path.join(DATA_DIR, 'train_X_white.npy'))
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin'))
    Xt = np.load(os.path.join(DATA_DIR, 'test_X_white.npy'))
    yt = read_labels(os.path.join(DATA_DIR, 'test_y.bin'))

    return X, y.astype(np.int32), Xt, yt.astype(np.int32)


def load_train_test_full_raw():
    X = read_all_images(os.path.join(DATA_DIR, 'train_X.bin')).astype(np.float32) / 255
    y = read_labels(os.path.join(DATA_DIR, 'train_y.bin'))
    Xt = read_all_images(os.path.join(DATA_DIR, 'test_X.bin')).astype(np.float32) / 255
    yt = read_labels(os.path.join(DATA_DIR, 'test_y.bin'))

    return X, y.astype(np.int32), Xt, yt.astype(np.int32)


def fold_idxs(fold):
    folds = open(os.path.join(DATA_DIR, 'fold_indices.txt')).readlines()
    return np.fromstring(folds[fold], dtype=int, sep=' ')


def augmentation(image, sz=96):
    image = tf.image.resize_image_with_crop_or_pad(image, sz+8, sz+8)
    offset = tf.random_uniform([1], maxval=16, dtype=tf.int32)[0]
    image = tf.random_crop(image, size=[sz+8-offset, sz+8-offset, 3])
    image = tf.image.resize_images(image, sz, sz)
    image.set_shape([sz, sz, 3])
    return image


def augmentation_scat_cropresize(image):
    return augmentation(image, sz=64)


def augmentation_scat(image):
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    gamma = tf.random_uniform([1], minval=0.6, maxval=1.4)
    image = image ** gamma
    return image


def augmentation_pad(image):
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

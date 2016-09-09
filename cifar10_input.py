import numpy as np
import os
import pickle
import tensorflow as tf


WHITENED_PKL = '/scratch/clear/abietti/data/cifar10_data/whitened.pkl'
WHITENED_CKN_MODEL = '/scratch/clear/abietti/results/ckn/cifar10white_py/layers_1.npy'

RAW_DATASET_FOLDER = '/home/clear/dwynen/ckn-dataset/cifar-10-batches-py/'
WHITENED_MAT = '/scratch/clear/abietti/data/cifar10_data/cifar10white.mat'


def params():
    return {
        'n_classes': 10,
        'lmbda': 6e-8,
        'lrs': [0.1, 0.3, 1.0, 3.0],
        'results_root': '/scratch/clear/abietti/results/ckn/cifar10white_py/accs',
    }


def params_scat():
    return {
        'n_classes': 10,
        'lmbda': 2e-8,
        'lrs': [0.1, 0.3, 1.0],
        'results_root': '/scratch/clear/abietti/results/ckn/cifar10white_py/accs',
    }


def unpickle_cifar(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def read_dataset_cifar10_raw(folder):
    """read the pickle files provided from 
    https://www.cs.toronto.edu/~kriz/cifar.html
    and returns all data in one numpy array for training and one for testing"""

    n_batch = 10000
    n_train = 5*n_batch
    n_test = n_batch

    # transpose to (n, h, w, channels) for tf image processing stuff
    Xtr = np.empty((n_train, 32, 32, 3), dtype=np.float32)
    Ytr = np.empty(n_train, dtype=np.int32)
    for i in range(1, 6):
        d = unpickle_cifar(os.path.join(folder, 'data_batch_{}'.format(i)))

        Xtr[(i-1)*n_batch:i*n_batch] = \
            d[b'data'].reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        Ytr[(i-1)*n_batch:i*n_batch] = d[b'labels']

    d = unpickle_cifar(os.path.join(folder, 'test_batch'))
    Xte = np.ascontiguousarray(d[b'data'].astype(np.float32).reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1))/255.0
    Yte = np.array(d[b'labels'], dtype=np.int32)
    return Xtr, Ytr, Xte, Yte


def read_dataset_cifar10_whitened(mat_file):
    """read cifar dataset from matlab whitened file."""
    from scipy.io import loadmat
    mat = loadmat(mat_file)

    def get_X(Xin):
        # HCWN -> NHWC
        return np.ascontiguousarray(Xin.astype(np.float32).reshape(32, 3, 32, -1).transpose(3, 0, 2, 1))

    return (get_X(mat['Xtr']), mat['Ytr'].ravel().astype(np.int32),
            get_X(mat['Xte']), mat['Yte'].ravel().astype(np.int32))


def load_cifar_pickle(pickled_file):
    X, y, Xt, yt = pickle.load(open(pickled_file, 'rb'))
    def getX(X):
        # already NHWC
        return np.ascontiguousarray(X.astype(np.float32))
    return getX(X), y.astype(np.int32), getX(Xt), yt.astype(np.int32)


def load_dataset_whitened():
    return load_cifar_pickle(WHITENED_PKL)


def load_dataset_raw():
    return read_dataset_cifar10_raw(RAW_DATASET_FOLDER)


def load_ckn_layers_whitened():
    return np.load(WHITENED_CKN_MODEL)


def get_scattering_params():
    from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
    filters, lw = multiresolution_filter_bank_morlet2d(32, J=3, L=8, sigma_phi=0.8, sigma_xi=0.8)
    m = 2
    return filters, m


def augmentation(image):
    # pad with zeros to make the image 36x36
    image = tf.image.resize_image_with_crop_or_pad(image, 36, 36)

    # extract random 32x32 crops
    image = tf.random_crop(image, size=[32, 32, 3])

    return image

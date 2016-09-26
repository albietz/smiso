import sys
import os
import shutil
import argparse
import json

import numpy as np
import scipy
import pickle

sys.path.append('/home/thoth/abietti/ckn_python/')
import test_ckn


def load_cifar(pickled_file):
    X, y, Xt, yt = pickle.load(open(pickled_file, 'rb'))
    def getX(X):
        # NHWC -> NCHW
        return np.ascontiguousarray(X.astype(np.float32).reshape(X.shape[0], 32, 32, 3).transpose(0, 3, 1, 2).reshape(-1, 96, 32))
    return getX(X), y, getX(Xt), yt


def read_dataset_cifar10_whitened(mat_file):
    """read cifar dataset from matlab whitened file."""
    from scipy.io import loadmat
    mat = loadmat(mat_file)

    def get_X(Xin):
        # HCWN -> NCHW
        return np.ascontiguousarray(Xin.astype(np.float32).reshape(32, 3, 32, -1).transpose(3, 1, 0, 2).reshape(-1, 96, 32))

    return get_X(mat['Xtr']), mat['Ytr'].ravel().astype(np.int32), get_X(mat['Xte']), mat['Yte'].ravel().astype(np.int32)


def load_mnist():
    from infimnist import _infimnist as imnist
    mnist = imnist.InfimnistGenerator()
    digits, labels = mnist.gen(np.arange(10000, 70000))  # training digits
    X = digits.reshape(-1, 28, 28).astype(np.float32)
    return X, labels


def load_stl10():
    Xt = np.load('/scratch/clear/abietti/data/stl10_binary/train_X_white.npy')
    Xu = np.load('/scratch/clear/abietti/data/stl10_binary/unlabeled_X_white.npy')
    _, H, W, C = Xt.shape
    return np.ascontiguousarray(np.vstack((Xt, Xu)).transpose(0, 3, 1, 2).reshape(-1, C*H, W))


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Trains a network and encodes the dataset with it.")
  
    parser.add_argument('--network',
        # default='network_cifar.json',
        # default='network_mnist.json',
        default='network_stl10.json',
        help='JSON file holding network hyperparameters')
    parser.add_argument('--dataset-file',
        default="/scratch/clear/abietti/data/cifar10_data/whitened.pkl",
        help="Pickle file with dataset.")
    parser.add_argument('--dataset-matfile',
        default="/scratch/clear/abietti/data/cifar10_data/cifar10white.mat",
        help="matlab file with dataset.")
    parser.add_argument('--results-root',
        # default='/scratch/clear/abietti/results/ckn/cifar10white_py/',
        # default='/scratch/clear/abietti/results/ckn/mnist_py/',
        default='/scratch/clear/abietti/results/ckn/stl10_white_py/',
        help='Root folder for results.')
    parser.add_argument('-N', '--n-patch-training', type=int,
        default=1000000,
        help='number of training patches to use for each layer')
    parser.add_argument('--cuda-device', type=int,
        default=0,
        help='cuda gpu id to use')
  
    args = parser.parse_args()
    os.makedirs(args.results_root, exist_ok=True)
 

    # X, y, Xt, yt = load_cifar(args.dataset_file)
    # X, y, Xt, yt = read_dataset_cifar10_whitened(args.dataset_matfile)
    # X, y = load_mnist()
    X = load_stl10()

    print('\nusing network architecture from {}'.format(args.network))
    network_file = os.path.join(args.results_root, 'network.json')
    print('making copy of json in {}'.format(network_file))
    shutil.copyfile(args.network, network_file)
    with open(network_file) as f:
        layers = json.load(f)

    layers = test_ckn.train_ckn(X, layers, args.results_root, args.cuda_device, args.n_patch_training)

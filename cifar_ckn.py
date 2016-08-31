import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf

import algos
import producer
import solvers
from ckn_encode_queue import CKNEncoder

sys.path.append('/home/thoth/abietti/ckn_python/')
import _ckn_cuda as ckn


cuda_device = 0
SEED = None
ENCODE_SIZE = 50000
CKN_BATCH_SIZE = 256

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
    Ytr = np.empty(n_train, dtype=np.float32)
    for i in range(1, 6):
        d = unpickle_cifar(os.path.join(folder, 'data_batch_{}'.format(i)))

        Xtr[(i-1)*n_batch:i*n_batch] = \
            d[b'data'].reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        Ytr[(i-1)*n_batch:i*n_batch] = d[b'labels']

    d = unpickle_cifar(os.path.join(folder, 'test_batch'))
    Xte = np.ascontiguousarray(d[b'data'].astype(np.float32).reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1))/255.0
    Yte = np.array(d[b'labels'], dtype=np.float32)
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


def load_mnist():
    from infimnist import _infimnist as imnist
    mnist = imnist.InfimnistGenerator()
    digits_train, labels_train = mnist.gen(np.arange(10000, 70000))  # training digits
    digits_test, labels_test = mnist.gen(np.arange(10000))  # test digits
    def getX(digits):
        return digits.reshape(-1, 28, 28, 1).astype(np.float32)
    return (getX(digits_train), labels_train.astype(np.int32),
            getX(digits_test), labels_test.astype(np.int32))


class Dataset(object):
    def __init__(self, dataset, augmentation=True, num_epochs=None,
                 num_threads=4, capacity=10000):
        self.num_epochs = num_epochs
        self.augmentation = augmentation
        self.train_data, self.train_labels, self.test_data, self.test_labels = dataset

        self.train_features = None
        self.test_features = None

        # create the queue
        self.images_initializer = tf.placeholder(dtype=self.train_data.dtype,
                                                 shape=self.train_data.shape)
        self.labels_initializer = tf.placeholder(dtype=self.train_labels.dtype,
                                                 shape=self.train_labels.shape)
        self.indexes_initializer = tf.placeholder(dtype=tf.int64,
                                                  shape=[self.train_data.shape[0]])
        self.input_images = tf.Variable(self.images_initializer, trainable=False, collections=[])
        self.input_labels = tf.Variable(self.labels_initializer, trainable=False, collections=[])
        self.input_indexes = tf.Variable(self.indexes_initializer, trainable=False, collections=[])

        image, label, index = producer.random_slice_input_producer(
                [self.input_images, self.input_labels, self.input_indexes],
                num_epochs=self.num_epochs, seed=SEED)
        if self.augmentation:
            image = self.process_train_image(image)

        # switch back to (channels, h, w) for ckn encoding
        image = tf.transpose(image, perm=[2, 0, 1])
        self.images, self.labels, self.indexes = tf.train.batch(
                [image, label, index], batch_size=ENCODE_SIZE, num_threads=num_threads, capacity=capacity)


    def init(self, sess):
        sess.run(self.input_images.initializer,
                 feed_dict={self.images_initializer: self.train_data})
        sess.run(self.input_labels.initializer,
                 feed_dict={self.labels_initializer: self.train_labels})
        sess.run(self.input_indexes.initializer,
                 feed_dict={self.indexes_initializer: range(self.train_labels.shape[0])})

    def init_test_features(self, layers, init_train=False):
        '''Initialize feature vectors on validation/test data from a given model.'''

        def eval_in_batches(data):
            N, H, W, C = data.shape
            X = data.transpose(0, 3, 1, 2).reshape(N, C*H, W)
            return ckn.encode_cudnn(np.ascontiguousarray(X), layers, cuda_device, CKN_BATCH_SIZE)

        if init_train:
            self.train_features = eval_in_batches(self.train_data)
        # self.validation_features = eval_in_batches(self.validation_data)
        self.test_features = eval_in_batches(self.test_data)

    def process_train_image(self, image):
        # pad with zeros to make the image 36x36
        image = tf.image.resize_image_with_crop_or_pad(image, 36, 36)

        # extract random 32x32 crops
        image = tf.random_crop(image, size=[32, 32, 3])

        return image


class DatasetIterator(object):
    def __init__(self, ds, layers):
        self.ds = ds
        self.layers = layers
        self.sess = tf.Session()
        self.ds.init(self.sess)
        if self.ds.augmentation:
            tf.train.start_queue_runners(sess=self.sess)

    def encode(self, image_data):
        return ckn.encode_cudnn(
                image_data.reshape(image_data.shape[0], 96, 32),
                layers, cuda_device, CKN_BATCH_SIZE)

    def run(self, num_epochs):
        n = self.ds.train_data.shape[0]        
        for step in range(n * num_epochs // ENCODE_SIZE):
            epoch = float(step) * ENCODE_SIZE / n
            if self.ds.augmentation:
                image_data, labels, indexes = self.sess.run(
                        [self.ds.images, self.ds.labels, self.ds.indexes])
                X = self.encode(image_data)
                yield (epoch, X, labels, indexes)
            else:
                indexes = np.random.randint(n, size=ENCODE_SIZE)
                yield (epoch, None, None, indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Runs solvers with data-augmentation.")

    parser.add_argument('--network',
      default='network.json',
      help='JSON file holding network hyperparameters')
    parser.add_argument('--dataset-folder',
      default="/home/clear/dwynen/ckn-dataset/cifar-10-batches-py/",
      help="Folder containing dataset files.")
    parser.add_argument('--dataset-matfile',
      default='data/cifar10white.mat',
      help='matlab file containing whitened dataset')
    parser.add_argument('--dataset-file',
      default='/scratch/clear/abietti/data/cifar10_data/whitened.pkl',
      help='pickle file containing (whitened) dataset')
    parser.add_argument('--layers-file',
      # default="/scratch/clear/abietti/results/ckn/cifar1/layers_1.npy",
      # default="/scratch/clear/abietti/results/ckn/cifar10white/cifar_ckn_model0.npy",
      default="/scratch/clear/abietti/results/ckn/cifar10white_py/layers_1.npy",
      # default="/scratch/clear/abietti/results/ckn/mnist_py/layers_1.npy",
      help="numpy model file containing matrices for all layers")
    parser.add_argument('--results-root',
      default='/scratch/clear/abietti/results/ckn/cifar10white/accs',
      help='Root folder for result files.')
    parser.add_argument('--results-file',
      default='tmp.pkl',
      help='Filename of results pickle file.')
    parser.add_argument('--encoded-ds-filename',
      default=None,
      help='store encoded dataset in this npy file')
    parser.add_argument('--augmentation',
      default=True,
      action='store_true',
      help='whether to enable data augmentation')
    parser.add_argument('--no-augmentation',
      dest='augmentation',
      action='store_false',
      help='whether to disable data augmentation')
    parser.add_argument('--normalize',
      default=True,
      action='store_true',
      help='whether to center and L2 normalize each training example')
    parser.add_argument('--no-normalize',
      dest='normalize',
      action='store_false',
      help='do not normalize')
    parser.add_argument('--cuda-device',
      default=0, type=int,
      help='CUDA GPU device number')
    parser.add_argument('--start-decay-step',
      default=20, type=int,
      help='step at which to start decaying the stepsize')
    parser.add_argument('--no-decay',
      default=False,
      action='store_true',
      help='whether to start decaying the stepsize at step specified at start-decay-step')
    parser.add_argument('--compute-loss',
      default=False,
      action='store_true',
      help='whether to compute train/test losses')

    args = parser.parse_args()
    print('augmentation:', args.augmentation, 'normalize:', args.normalize,
          'no-decay:', args.no_decay)

    cuda_device = args.cuda_device
    layers = np.load(args.layers_file)

    with tf.device('/cpu:0'):
        # ds = Dataset(load_cifar_pickle('/scratch/clear/abietti/data/cifar10_data/whitened.pkl'),
        ds = Dataset(read_dataset_cifar10_whitened(args.dataset_matfile),
                     augmentation=args.augmentation)
    # leave here to avoid stream executor issues by creating session
    engine = DatasetIterator(ds, layers)
    init_train = args.compute_loss or not ds.augmentation
    ds.init_test_features(layers, init_train=init_train)

    n_classes = 10
    Xtest = ds.test_features.astype(solvers.dtype)
    if args.normalize:
        solvers.center(Xtest)
        solvers.normalize(Xtest)
    ytest = ds.test_labels
    if init_train:
        Xtrain = ds.train_features.astype(solvers.dtype)
        if args.normalize:
            solvers.center(Xtrain)
            solvers.normalize(Xtrain)
        ytrain = ds.train_labels

        if args.encoded_ds_filename:
            np.save(args.encoded_ds_filename, [{'Xtr': Xtrain, 'ytr': ds.train_labels, 'Xte': Xtest, 'yte': ds.test_labels}])

    dim = Xtest.shape[1]
    n = ds.train_data.shape[0]

    loss = algos.LogisticLoss()
    lmbda = 6e-8
    # lmbda = 0.001
    solver_list = [
        solvers.MISOOneVsRest(n_classes, dim, n, lmbda=lmbda, loss=loss.name.encode('utf-8')),
    ]
    solver_params = [dict(name='miso_onevsrest', lmbda=lmbda, loss=loss.name)]
    # adjust miso step-size if needed (with L == 1)
    solver_list[0].decay(min(1, lmbda * n / (1 - lmbda)))

    lrs = [0.1, 0.3, 1.0, 3.0]
    print('lrs:', lrs)
    for lr in lrs:
        solver_list.append(solvers.SGDOneVsRest(
                n_classes, dim, lr=lr, lmbda=lmbda, loss=loss.name.encode('utf-8')))
        solver_params.append(dict(name='sgd_onevsrest', lr=lr, lmbda=lmbda, loss=loss.name))

    n_algos = len(solver_list)
    start_time = time.time()

    test_accs = []
    train_losses = []
    test_losses = []
    for step, (e, Xdata, labels, idxs) in enumerate(engine.run(500)):
        t0 = time.time()
        if ds.augmentation:
            X = Xdata.astype(solvers.dtype)
            y = labels
            if args.normalize:
                solvers.center(X)
                solvers.normalize(X)
        else:
            X = Xtrain
            y = ytrain
        times = []
        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []
        for alg, solver in enumerate(solver_list):
            if not args.no_decay and step == args.start_decay_step:
                # print('starting stepsize decay')
                # solver.start_decay()
                print('decaying stepsize')
                solver.decay(0.5)

            t1 = time.time()
            if ds.augmentation:
                solver.iterate(X, y, idxs)
            else:
                solver.iterate_indexed(X, y, idxs)
            times.append(time.time() - t1)
            acc_train.append((solver.predict(X) == y).mean())
            acc_test.append((solver.predict(Xtest) == ytest).mean())
            if args.compute_loss:
                loss_train.append(solver.compute_loss(Xtrain, ytrain))
                loss_test.append(solver.compute_loss(Xtest, ytest))

        print('epoch', e)
        print('train batch acc', acc_train)
        print('test acc', acc_test)
        print('train loss', loss_train)
        print('test loss', loss_test)
        t = time.time()
        print('elapsed time:', t - start_time,
              'training/evaluation elapsed time:', t - t0,
              'iterate times:', times)
        start_time = t
        test_accs.append(acc_test)
        if args.compute_loss:
            train_losses.append(loss_train)
            test_losses.append(loss_test)
        sys.stdout.flush()

    pickle.dump({'params': solver_params, 'test_accs': test_accs,
                 'train_losses': train_losses, 'test_losses': test_losses},
                open(os.path.join(args.results_root, args.results_file), 'wb'))


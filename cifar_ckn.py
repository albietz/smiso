import argparse
import json
import os
import pickle
import time
import numpy as np
import tensorflow as tf

import algos
import producer
import solvers
from ckn import _ckn_cuda as ckn


cuda_device = 0
SEED = None
ENCODE_SIZE = 10000
CKN_BATCH_SIZE = 256

def unpickle_cifar(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def read_dataset_cifar10(folder):
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


class Dataset(object):
    def __init__(self, dataset_folder, augmentation=True, num_epochs=None,
                 num_threads=10, capacity=256):
        self.num_epochs = num_epochs
        self.train_data, self.train_labels, self.test_data, self.test_labels = \
                read_dataset_cifar10(dataset_folder)

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
        if augmentation:
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
            return ckn.encode_cudnn(data.reshape(data.shape[0], 96, 32), layers, cuda_device, CKN_BATCH_SIZE)

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
        tf.train.start_queue_runners(sess=self.sess)

    def encode(self, image_data):
        return ckn.encode_cudnn(
                image_data.reshape(image_data.shape[0], 96, 32),
                layers, cuda_device, CKN_BATCH_SIZE)

    def run(self, num_epochs):
        n = self.ds.train_data.shape[0]        
        for step in range(n * num_epochs // ENCODE_SIZE):
            image_data, labels, indexes = self.sess.run(
                    [self.ds.images, self.ds.labels, self.ds.indexes])
            epoch = float(step) * ENCODE_SIZE / n
            X = self.encode(image_data)
            yield (epoch, X, labels, indexes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Trains a network and encodes the dataset with it.")

    parser.add_argument('--network',
      default='network.json',
      help='JSON file holding network hyperparameters')
    parser.add_argument('--dataset-folder',
      default="/home/clear/dwynen/ckn-dataset/cifar-10-batches-py/",
      help="Folder containing dataset files.")
    parser.add_argument('--layers-file',
      default="/scratch/clear/abietti/results/ckn/cifar1/layers_1.npy",
      help="numpy model file containing matrices for all layers")
    parser.add_argument('--results-root',
      default='/scratch/clear/abietti/results/ckn_incr',
      help='Root folder for results. Will make a subfolder there based on $tag')
    parser.add_argument('--cuda-device',
      default=0, type=int,
      help='CUDA GPU device number')

    args = parser.parse_args()

    cuda_device = args.cuda_device
    layers = np.load(args.layers_file)

    with tf.device('/cpu:0'):
        ds = Dataset(args.dataset_folder)
    sess = tf.Session()  # prevent stream executor issue
    ds.init_test_features(layers)

    target_label = 2
    Xtest = ds.test_features.astype(np.float64)
    ytest = (ds.test_labels == target_label).astype(np.float64)

    dim = Xtest.shape[1]
    n = ds.train_data.shape[0]

    loss = algos.LogisticLoss()
    lmbda = 0.001
    solver_list = [
            solvers.SGD(dim, lr=0.1, lmbda=lmbda, loss=loss.name.encode('utf-8')),
            solvers.MISO(dim, n, lmbda=lmbda, loss=loss.name.encode('utf-8')),
        ]
    engine = DatasetIterator(ds, layers)
    start_time = time.time()
    for step, (e, Xdata, labels, idxs) in enumerate(engine.run(5)):
        X = Xdata.astype(np.float64)
        y = (labels == target_label).astype(np.float64)

        if step % 3 == 0:
            for solver in solver_list:
                solver.decay()

        losses = []
        accuracies = []  # auc/ap?
        for solver in solver_list:
            solver.iterate(X, y, idxs)
            preds = Xtest.dot(solver.w)
            losses.append(loss.compute(preds, ytest).mean())

        print('epoch', e, 'test losses', losses)
        t = time.time()
        print('elapsed time:', t - start_time)
        start_time = t


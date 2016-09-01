import numpy as np
import tensorflow as tf
import sys

from infimnist import infimnist_queue
import producer

sys.path.append('/home/thoth/abietti/ckn_python/')
import _ckn_cuda as ckn


class CKNDatasetBase(object):
    def __init__(self, dataset, augmentation=True, cuda_device=0, ckn_batch_size=256):
        self.augmentation = augmentation
        self.cuda_device = cuda_device
        self.ckn_batch_size = ckn_batch_size
        self.train_data, self.train_labels, self.test_data, self.test_labels = dataset

        self.train_features = None
        self.test_features = None

    def init_test_features(self, layers, init_train=False):
        '''Initialize feature vectors on validation/test data from a given model.'''

        def eval_in_batches(data):
            N, H, W, C = data.shape
            X = data.transpose(0, 3, 1, 2).reshape(N, C*H, W)
            return ckn.encode_cudnn(np.ascontiguousarray(X), layers,
                                    self.cuda_device, self.ckn_batch_size)

        if init_train:
            self.train_features = eval_in_batches(self.train_data)
        # self.validation_features = eval_in_batches(self.validation_data)
        self.test_features = eval_in_batches(self.test_data)

    def init(self, sess, coord=None):
        pass

    def close(self, sess, coord=None):
        pass


class CKNDataset(CKNDatasetBase):
    def __init__(self, dataset, augmentation=True, augm_fn=None, num_epochs=None,
                 num_threads=4, batch_size=10000, capacity=20000, seed=None,
                 cuda_device=0, ckn_batch_size=256):
        super(CKNDataset, self).__init__(
                dataset, augmentation=augmentation,
                cuda_device=cuda_device, ckn_batch_size=ckn_batch_size)

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
                num_epochs=num_epochs, seed=seed, capacity=10000)
        if self.augmentation and augm_fn is not None:
            image = augm_fn(image)

        self.images, self.labels, self.indexes = tf.train.batch(
                [image, label, index], batch_size=batch_size, num_threads=num_threads, capacity=capacity)
        # switch from NHWC to NCHW for ckn encoding
        self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])

    def init(self, sess, coord=None):
        sess.run(self.input_images.initializer,
                 feed_dict={self.images_initializer: self.train_data})
        sess.run(self.input_labels.initializer,
                 feed_dict={self.labels_initializer: self.train_labels})
        sess.run(self.input_indexes.initializer,
                 feed_dict={self.indexes_initializer: range(self.train_labels.shape[0])})


class CKNInfimnistDataset(CKNDatasetBase):
    def __init__(self, dataset, batch_size=10000, capacity=20000,
                 cuda_device=0, ckn_batch_size=256):
        super(CKNInfimnistDataset, self).__init__(
                dataset, augmentation=True,
                cuda_device=cuda_device, ckn_batch_size=ckn_batch_size)

        self.producer = infimnist_queue.InfimnistProducer(
                batch_size=batch_size, gen_batch_size=batch_size, capacity=capacity)

        self.images, self.labels, self.indexes = \
                self.producer.digits, self.producer.labels, self.producer.indexes
        # switch from NHWC to NCHW for ckn encoding
        self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])

    def init(self, sess, coord=None):
        self.producer.start_queue(sess, coord)

    def close(self, sess, coord=None):
        self.producer.join(sess, coord)

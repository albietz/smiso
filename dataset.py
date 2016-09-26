from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sys
import tensorflow as tf
import time

from infimnist import infimnist_queue
import producer


class DatasetBase(object):
    def __init__(self, dataset, augmentation=True):
        self.augmentation = augmentation
        self.train_data, self.train_labels, self.test_data, self.test_labels = dataset

        self.train_features = None
        self.test_features = None

    def init_features(self, model_type='ckn', model_params=None, init_train=False):
        '''Initialize feature vectors on validation/test data from a given model.'''

        model_params = model_params or {}

        if model_type == 'ckn':
            sys.path.append('/home/thoth/abietti/ckn_python/')
            import _ckn_cuda as ckn

            layers = model_params['layers']
            cuda_devices = model_params['cuda_devices']
            ckn_batch_size = model_params['ckn_batch_size']

            def eval_in_batches(data, cuda_device):
                N, H, W, C = data.shape
                X = data.transpose(0, 3, 1, 2).reshape(N, C*H, W)
                return ckn.encode_cudnn(np.ascontiguousarray(X), layers,
                                        cuda_device, ckn_batch_size)

            with ThreadPoolExecutor(max_workers=len(cuda_devices)) as executor:
                test_future = executor.submit(eval_in_batches, self.test_data, cuda_devices[0])
                if init_train:
                    train_future = executor.submit(eval_in_batches, self.train_data,
                                                   cuda_devices[1 % len(cuda_devices)])
                    self.train_features = train_future.result()
                self.test_features = test_future.result()

        elif model_type == 'scattering':
            encoder = model_params['encoder']
            print('encoding...')
            t = time.time()
            if init_train:
                self.train_features = encoder.encode_nhwc(self.train_data)
            self.test_features = encoder.encode_nhwc(self.test_data)
            print('done encoding. time elapsed', time.time() - t)

    def init(self, sess, coord=None):
        pass

    def close(self, sess, coord=None):
        pass


class Dataset(DatasetBase):
    def __init__(self, dataset, augmentation=True, augm_fn=None, num_epochs=None,
                 producer_type='random_index', num_threads=4, batch_size=10000,
                 capacity=20000, seed=None):
        super().__init__(dataset, augmentation=augmentation)

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

        if producer_type == 'random_index':
            image, label, index = producer.random_slice_input_producer(
                    [self.input_images, self.input_labels, self.input_indexes],
                    num_epochs=num_epochs, seed=seed, capacity=10000)
        elif producer_type == 'epoch':
            image, label, index = tf.train.slice_input_producer(
                    [self.input_images, self.input_labels, self.input_indexes],
                    num_epochs=num_epochs, shuffle=False, seed=seed, capacity=10000)
        else:
            print('wrong producer type', producer_type)
            sys.exit(0)

        if self.augmentation and augm_fn is not None:
            image = augm_fn(image)

        self.images, self.labels, self.indexes = tf.train.batch(
                [image, label, index], batch_size=batch_size, num_threads=num_threads, capacity=capacity)
        # switch from NHWC to NCHW for encoding
        self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])

    def init(self, sess, coord=None):
        sess.run(self.input_images.initializer,
                 feed_dict={self.images_initializer: self.train_data})
        sess.run(self.input_labels.initializer,
                 feed_dict={self.labels_initializer: self.train_labels})
        sess.run(self.input_indexes.initializer,
                 feed_dict={self.indexes_initializer: range(self.train_labels.shape[0])})


class InfimnistDataset(DatasetBase):
    def __init__(self, dataset, batch_size=10000, capacity=20000):
        super().__init__(dataset, augmentation=True)

        self.producer = infimnist_queue.InfimnistProducer(
                batch_size=batch_size, gen_batch_size=batch_size, capacity=capacity)

        self.images, self.labels, self.indexes = \
                self.producer.digits, self.producer.labels, self.producer.indexes
        # switch from NHWC to NCHW for ckn/scattering encoding
        self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])

    def init(self, sess, coord=None):
        self.producer.start_queue(sess, coord)

    def close(self, sess, coord=None):
        self.producer.join(sess, coord)

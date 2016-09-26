import sys
import time
import numpy as np
import tensorflow as tf


class DatasetIterator(object):
    def __init__(self, ds, model_type='ckn', model_params=None, encode_size=50000):
        self.ds = ds
        self.encoder = None
        self.encode_size = encode_size
        if self.ds.augmentation:
            if model_type == 'ckn':
                from ckn_queued_encoder import CKNQueuedEncoder
                self.encoder = CKNQueuedEncoder(
                        [self.ds.images, self.ds.labels, self.ds.indexes],
                        self.ds.test_features.shape[1],
                        model_params['layers'], batch_size=encode_size,
                        cuda_devices=model_params['cuda_devices'],
                        ckn_batch_size=model_params['ckn_batch_size'])
            elif model_type == 'scattering':
                from scattering_queued_encoder import ScatteringQueuedEncoder
                self.encoder = ScatteringQueuedEncoder(
                    [self.ds.images, self.ds.labels, self.ds.indexes],
                    self.ds.test_features.shape[1], encoder=model_params['encoder'],
                    batch_size=encode_size)
            else:
                print('bad model type for data augmentation:', model_type)
                sys.exit(0)

        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.ds.init(self.sess, self.coord)
        if self.ds.augmentation:
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.encoder.start_queue(self.sess, self.coord)

    def run(self, num_epochs):
        n = self.ds.train_data.shape[0]        
        n_steps = n * num_epochs // self.encode_size
        for step in range(n_steps):
            epoch = float(step) * self.encode_size / n
            if self.ds.augmentation:
                X, labels, indexes = self.sess.run(
                        [self.encoder.encoded, self.encoder.labels, self.encoder.indexes])
                yield (epoch, X, labels, indexes)
            else:
                indexes = np.random.randint(n, size=encode_size)
                yield (epoch, None, None, indexes)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.ds.close(self.sess, self.coord)
        self.encoder.join(self.sess, self.coord)
        self.sess.close()

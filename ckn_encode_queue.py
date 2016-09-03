import sys
import threading
import tensorflow as tf

sys.path.append('/home/thoth/abietti/ckn_python/')
import _ckn_cuda as ckn


class CKNEncoder(object):
    def __init__(self, input_ops, encode_dim, ckn_layers, batch_size=1000,
                 cuda_devices=None, ckn_batch_size=256):
        self.images, self.labels_in, self.indexes_in = input_ops
        self.layers = ckn_layers
        self.batch_size = batch_size
        self.encode_dim = encode_dim
        self.cuda_devices = cuda_devices or [0]
        self.ckn_batch_size = ckn_batch_size
        self.threads = []

        self.encoded_placeholder = tf.placeholder(tf.float32,
                                                  shape=[self.batch_size, self.encode_dim])
        self.labels_placeholder = tf.placeholder(self.labels_in.dtype, shape=self.labels_in.get_shape())
        self.indexes_placeholder = tf.placeholder(self.indexes_in.dtype, shape=self.indexes_in.get_shape())
        # enough capacity for each GPU
        self.q = tf.FIFOQueue(capacity=len(self.cuda_devices) * self.batch_size,
                              dtypes=[tf.float32, self.labels_in.dtype, self.indexes_in.dtype],
                              shapes=[[self.encode_dim], [], []])

        self.enqueue_op = self.q.enqueue_many([self.encoded_placeholder,
                                               self.labels_placeholder,
                                               self.indexes_placeholder])

        # dequeue one batch (same batch size as enqueue)
        self.encoded, self.labels, self.indexes = self.q.dequeue_many(self.batch_size)

    def encode_thread(self, sess, coord, cuda_device):
        while coord is None or not coord.should_stop():
            try:
                print('GETTING INPUT DATA')
                images, labels, indexes = sess.run([self.images, self.labels_in, self.indexes_in])
                N, C, H, W = images.shape
                print('ENCODING INPUT DATA')
                X = ckn.encode_cudnn(images.reshape(N, C*H, W), self.layers,
                                     cuda_device, self.ckn_batch_size)

                if coord is not None and coord.should_stop():
                    break
                print('ENQUEUING INPUT DATA')
                sess.run(self.enqueue_op, feed_dict={self.encoded_placeholder: X,
                                                     self.labels_placeholder: labels,
                                                     self.indexes_placeholder: indexes})
            except tf.errors.CancelledError:
                break

    def start_queue(self, sess, coord=None):
        assert not self.threads, 'queue already started!'
        for cuda_device in self.cuda_devices:
            thread = threading.Thread(target=self.encode_thread, args=(sess, coord, cuda_device))
            self.threads.append(thread)
            thread.start()

    def join(self, sess, coord=None):
        # sess.run(self.q.close(cancel_pending_enqueues=True))
        if self.threads:
            if coord is None:
                for thread in self.threads:
                    thread.join()
            else:
                coord.join(self.threads)

import threading
import tensorflow as tf


class QueuedEncoder(object):
    def __init__(self, input_ops, encode_dim, batch_size=1000, batch_capacity=1):
        self.images, self.labels_in, self.indexes_in = input_ops
        self.batch_size = batch_size
        # the capacity of the queue will be batch_size * batch_capacity.
        # batch_capacity also defines the number of concurrent encoding threads
        # e.g. one per GPU for CKNs
        self.batch_capacity = batch_capacity
        self.encode_dim = encode_dim
        self.threads = []

        self.encoded_placeholder = tf.placeholder(tf.float32,
                                                  shape=[self.batch_size, self.encode_dim])
        self.labels_placeholder = tf.placeholder(self.labels_in.dtype, shape=self.labels_in.get_shape())
        self.indexes_placeholder = tf.placeholder(self.indexes_in.dtype, shape=self.indexes_in.get_shape())
        self.q = tf.FIFOQueue(capacity=batch_capacity * self.batch_size,
                              dtypes=[tf.float32, self.labels_in.dtype, self.indexes_in.dtype],
                              shapes=[[self.encode_dim], [], []])

        self.enqueue_op = self.q.enqueue_many([self.encoded_placeholder,
                                               self.labels_placeholder,
                                               self.indexes_placeholder])

        # dequeue one batch (same batch size as enqueue)
        self.encoded, self.labels, self.indexes = self.q.dequeue_many(self.batch_size)

    def encode_thread(self, sess, coord, thread_index):
        while coord is None or not coord.should_stop():
            try:
                print('GETTING INPUT DATA')
                images, labels, indexes = sess.run([self.images, self.labels_in, self.indexes_in])
                print('ENCODING IMAGES')
                X = self.encode_images(images, thread_index)

                if coord is not None and coord.should_stop():
                    break
                print('ENQUEUING INPUT DATA')
                sess.run(self.enqueue_op, feed_dict={self.encoded_placeholder: X,
                                                     self.labels_placeholder: labels,
                                                     self.indexes_placeholder: indexes})
            except tf.errors.CancelledError:
                break

    def encode_images(self, images, thread_index):
        raise NotImplementedError()

    def start_queue(self, sess, coord=None):
        assert not self.threads, 'queue already started!'
        for i in range(self.batch_capacity):
            thread = threading.Thread(target=self.encode_thread, args=(sess, coord, i))
            self.threads.append(thread)
            thread.daemon = True
            thread.start()

    def join(self, sess, coord=None, cancel_pending_enqueues=True):
        sess.run(self.q.close(cancel_pending_enqueues=cancel_pending_enqueues))
        if self.threads:
            if coord is None:
                for thread in self.threads:
                    thread.join()
            else:
                coord.join(self.threads)

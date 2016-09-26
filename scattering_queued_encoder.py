import tensorflow as tf
import time

import queued_encoder


class ScatteringQueuedEncoder(queued_encoder.QueuedEncoder):
    def __init__(self, input_ops, encode_dim, encoder, batch_size=1000):
        super().__init__(input_ops, encode_dim, batch_size=batch_size,
                         batch_capacity=1)  # single thread
        self.encoder = encoder

    def encode_images(self, images, thread_index):
        t = time.time()
        X = self.encoder.encode_nchw(images)
        print('encoding time', time.time() - t)
        return X

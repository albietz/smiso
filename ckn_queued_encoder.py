import sys
import tensorflow as tf

import queued_encoder
sys.path.append('/home/thoth/abietti/ckn_python/')
import _ckn_cuda as ckn


class CKNQueuedEncoder(queued_encoder.QueuedEncoder):
    def __init__(self, input_ops, encode_dim, ckn_layers, batch_size=1000,
                 cuda_devices=None, ckn_batch_size=256):
        super().__init__(input_ops, encode_dim, batch_size=batch_size,
                         batch_capacity=len(cuda_devices or [0]))
        self.layers = ckn_layers
        self.cuda_devices = cuda_devices or [0]
        self.ckn_batch_size = ckn_batch_size

    def encode_images(self, images, thread_index):
        cuda_device = self.cuda_devices[thread_index]
        N, C, H, W = images.shape
        return ckn.encode_cudnn(images.reshape(N, C*H, W), self.layers,
                                cuda_device, self.ckn_batch_size)

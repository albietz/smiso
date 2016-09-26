import multiprocessing
import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor
from skimage.color.colorconv import rgb2yuv
from skimage.feature.scattering import scattering


USE_YUV = True
pool = None


def create_pool(num_workers):
    global pool
    pool = multiprocessing.Pool(num_workers)


def close_pool():
    global pool
    if pool is not None:
        pool.close()


# for ProcessPoolExecutor
def process_worker(im, dim, filters, m, batch_size):
    N, C, H, W = im.shape
    if USE_YUV:
        im = rgb2yuv(im.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
    X = np.empty((im.shape[0], dim), dtype=np.float32)
    for i in range(0, im.shape[0], batch_size):
        b = min(batch_size, im.shape[0] - i)
        s, _, _ = scattering(im[i:i + b].reshape(-1, H, W),
                             filters, m=m)
        X[i:i + b] = s.reshape(b, -1)
    return X


class ScatteringEncoder(object):
    def __init__(self, filters, m=2, batch_size=500):
        self.filters = filters
        self.m = m
        self.batch_size = batch_size
        self.dim = None  # feature dimension, to be inferred

    def encode_nchw(self, images):
        global pool
        N, C, H, W = images.shape

        # infer dimension for allocation
        if self.dim is None:
            s, _, _ = scattering(images[0,:1], self.filters, m=self.m)
            s = s.reshape(s.shape[0], -1)
            self.dim = C * s.shape[1]

        # with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
        num_workers = pool._processes
        res = []
        n_per_worker = N // num_workers + 1
        for i in range(0, N, n_per_worker):
            res.append(pool.apply_async(
                process_worker, (images[i:i+n_per_worker],
                self.dim, self.filters, self.m, self.batch_size)))

        return np.vstack([r.get() for r in res])

    def encode_nhwc(self, images):
        return self.encode_nchw(images.transpose(0, 3, 1, 2))


if __name__ == '__main__':
    from cifar10_input import load_dataset_raw, get_scattering_params

    im, _, _, _ = load_dataset_raw()
    filters, m = get_scattering_params()

    encoder = ScatteringEncoder(filters, m, num_workers=56)

    t = time.time()
    X = encoder.encode_nhwc(im)
    print(time.time() - t, 'seconds elapsed, shape', im.shape, X.shape)

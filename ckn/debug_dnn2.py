#! /usr/bin/env python3

import numpy as np
import numpy.matlib
import scipy
import scipy.io


# read matlab model
def read_matlab_ckn_model(filename):
    assert filename.endswith('.mat')
    mat = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
    layers_mat = mat['model'].layer
    layers = []
    if not isinstance(layers_mat, np.ndarray):
        # only have a single layer
        layers_mat = [layers_mat]
    for k, l in enumerate(layers_mat):
        print('\nLayer {}'.format(k+1))
        items = l.__dict__.items()
        items = sorted(items, key = lambda t: t[0])
        for k, v in items:
          if k.startswith('_'): continue
          print('{}: {}'.format(k, v.shape if isinstance(v, np.ndarray) else v))
        l.zero_padding = bool(l.zero_padding)
        print('zero_padding: {}'.format(l.zero_padding))
        l.new_subsampling = bool(l.new_subsampling)
        # TODO: do this for all fields
        # and don't fail when one of them is not an array
        l.b = l.b.flatten().astype(np.float32)
        # l.W = l.W.T.astype(np.float32)
        # l.W.shape = l.W.shape[0], -1

        # correct dimensions
        nmaps = l.W.shape[0] // (l.npatch * l.npatch)
        l.W = l.W.astype(np.float32).reshape(nmaps, l.npatch, l.npatch, -1).transpose(3, 0, 1, 2)
        l.W = l.W.reshape(l.W.shape[0], -1)
        l.W2 = l.W2.T.astype(np.float32)
        l.W2.shape = l.W2.shape[0], -1
        # print(l.Wfilt)
        # l.Wfilt = l.Wfilt.T.astype(np.float32)
        # l.Wfilt.shape = l.Wfilt.shape[0], -1

        # l.W3 = l.W3.T.astype(np.float32)
        # l.Wfilt = l.Wfilt.T.astype(np.float32)

        # in the c++ version the filters are computed and then multiplied by -(1/sigma**2)
        # during encoding, the filters and bias are computed as (W.T.dot(X) + b)
        # in the nips16 paper we use alpha * (W.T.dot(X) - 1)
        # so to get the same result, we modify the filters and bias here.

        # need filters in correct shape and order, c++ only stores a matrix
        # l.W.shape = (l.nfilters, l.npatch, l.npatch, -1)
        # assert l.W.shape[3] == layers_mat[k-1].nfilters if k > 0 else 3
        # l.W = np.ascontiguousarray(l.W.transpose(1, 2, 3, 0))

        # l.alpha = 1 / l.sigma**2.0
        # diff = np.abs(l.alpha + l.b[0])
        # assert diff < 1e-5
        # l.filters = l.W
        # l.A = l.W2
        # del l.W2
        layers.append({
            'sigma': l.sigma,
            'type_layer': l.type_layer,
            'W': l.W,
            'W2': l.W2,
            'W3': l.W3,
            'b': l.b,
            'zero_padding': l.zero_padding,
            'npatch': l.npatch,
            'nfilters': l.nfilters,
            'subsampling': l.subsampling,
            'new_subsampling': l.new_subsampling,
            })

    print()
    return layers

if __name__ == '__main__':
    import _ckn_cuda as ckn
    layers = read_matlab_ckn_model('/home/clear/dwynen/tmp/model_cifar-10_3_2_0_2_5_0_256_1024_0_0_1.000000e-02_1.000000e-02_1.000000e-02_0_2_0_0_2.mat')
    mat = scipy.io.loadmat('/home/lear/dwynen/code/ckn_cpp/tmp.mat')
    X = mat['X'].transpose(2, 1, 0)[:512]
    X = np.ascontiguousarray(np.tile(X.astype(np.float32), (10, 1, 1)))
    print(X.shape)
    print(X.dtype)
    print(X.flags)
    psiml = scipy.io.loadmat('/home/lear/dwynen/code/ckn_cpp/tmpout.mat')['psi'].T

    print('\nencoding on CPU')
    psipycpu = ckn.encode_cpu(X.copy(), layers, threads=0)
    print('relative error norm cpu:  {}'.format(np.linalg.norm(psipycpu-psiml)/np.linalg.norm(psiml)))

    print('\nencoding on GPU')
    psipy = ckn.encode_cudnn(X.copy(), layers, 0, 128)

    print('psi    (ml): {}'.format(psiml.shape))
    print('psi    (py): {}'.format(psipycpu.shape))
    print('psicuda:     {}'.format(psipy.shape))

    print('relative error norm cuda: {}'.format(np.linalg.norm(psipy-psiml)/np.linalg.norm(psiml)))

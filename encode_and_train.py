import argparse
import solvers
import numpy as np
import time
import sys

import cifar10_input
import mnist_input
import stl10_input

import _ckn_cuda as ckn
import miso


def encode_ckn(images, layers, cuda_device, ckn_batch_size):
    N, H, W, C = images.shape
    X = images.transpose(0, 3, 1, 2).reshape(N, C*H, W)
    return ckn.encode_cudnn(np.ascontiguousarray(X), layers,
                            cuda_device, ckn_batch_size)


def load_encoded(expt, cuda_device=0):
    print('loading experiment data for', expt)
    if expt == 'cifar10':
        model = cifar10_input.load_ckn_layers_whitened()
        data = cifar10_input.load_dataset_whitened()
        expt_params = cifar10_input.params()
    elif expt.startswith('mnist'):
        num_transformations = 0  # no augmentation by default
        if len(expt) > 5:
            num_transformations = int(expt[5:])
        model = mnist_input.load_ckn_layers()
        data = mnist_input.load_dataset(num_transformations)
        expt_params = mnist_input.params()
    elif expt.startswith('stl10'):
        # format: stl10_<fold><t[est]/v[al]>
        # e.g. stl10_3v for training with fold 3 and testing with
        # the rest of the training set as validation set
        fold = int(expt[6])
        if expt[7] == 't':
            data = stl10_input.load_train_test_white(fold)
        else:
            data = stl10_input.load_train_val_white(fold)
        model = stl10_input.load_ckn_layers_whitened()
        expt_params = stl10_input.params()
    else:
        print('experiment', expt, 'not supported!')
        sys.exit(0)

    images_train, y, images_test, yt = data
    ckn_batch_size = expt_params.get('ckn_batch_size', 256)

    print('encoding...')
    return (encode_ckn(images_train, model, cuda_device, ckn_batch_size), y.astype(np.float32),
            encode_ckn(images_test, model, cuda_device, ckn_batch_size), yt.astype(np.float32))


def load_encoded_scat(expt, num_workers=10):
    print('loading experiment data for', expt)
    if expt == 'cifar10':
        # data = cifar10_input.load_dataset_whitened()
        data = cifar10_input.load_dataset_raw()
        filters, m = cifar10_input.get_scattering_params()
    elif expt.startswith('stl10'):
        # format: stl10_<fold><t[est]/v[al]>
        # e.g. stl10_3v for training with fold 3 and testing with
        # the rest of the training set as validation set
        fold = int(expt[6])
        if expt[7] == 't':
            data = stl10_input.load_train_test_raw(fold)
        else:
            data = stl10_input.load_train_val_raw(fold)
        def pad(im):
            padded = np.zeros((im.shape[0], 128, 128, im.shape[3]), dtype=np.float32)
            padded[:,16:112,16:112,:] = im
            return padded
        data = (pad(data[0]), data[1], pad(data[2]), data[3])
        filters, m = stl10_input.get_scattering_params()
    else:
        print('experiment', expt, 'not supported!')
        sys.exit(0)

    images_train, y, images_test, yt = data

    from scattering_encoder import create_pool, close_pool, ScatteringEncoder
    create_pool(num_workers)
    encoder = ScatteringEncoder(filters, m)
    print('encoding...')
    t = time.time()
    X = encoder.encode_nhwc(images_train)
    Xt = encoder.encode_nhwc(images_test)
    print('done encoding. time elapsed', time.time() - t)
    close_pool()
    return (X, y.astype(np.float32), Xt, yt.astype(np.float32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Trains a network and encodes the dataset with it.")

    parser.add_argument('--experiment',
      default='cifar10',
      help='experiment (data, model)')
    parser.add_argument('--model-type',
      default='ckn',
      help='ckn or scattering')
    parser.add_argument('--normalize',
      default=True,
      action='store_true',
      help='whether to center and L2 normalize each training example')
    parser.add_argument('--no-normalize',
      dest='normalize',
      action='store_false',
      help='do not normalize')
    parser.add_argument('--cuda-device',
      default=0, type=int,
      help='CUDA GPU device number')
    parser.add_argument('--num-workers',
      default=10, type=int,
      help='number of parallel processes for scattering encoding')
    parser.add_argument('--verbose',
      default=False, action='store_true',
      help='make miso training verbose')

    args = parser.parse_args()

    if args.model_type == 'ckn':
        X, y, Xt, yt = load_encoded(args.experiment, cuda_device=args.cuda_device)
    elif args.model_type == 'scattering':
        X, y, Xt, yt = load_encoded_scat(args.experiment, num_workers=args.num_workers)
    else:
        print('model-type must be ckn or scattering')
        sys.exit(0)

    if args.normalize:
        print('normalizing...')
        solvers.center(X)
        solvers.normalize(X)
        solvers.center(Xt)
        solvers.normalize(Xt)

    # lmbdas = 2. ** np.arange(-10, -25, -2)
    # lmbdas = 2. ** np.arange(-7, -20, -2)
    # lmbdas = 2. ** np.arange(-20, -31, -2)
    lmbdas = [6e-4, 4e-4, 2e-4, 9e-5, 4e-5]
    train_accs = []
    test_accs = []
    lmbda_best = None
    acc_best = None
    for lmbda in lmbdas:
        if not args.verbose:
            print('lambda = {:10e}'.format(lmbda), end=' ', flush=True)
        clf = miso.MisoClassifier(lmbda, eps=1e-4, max_iterations=500 * X.shape[0],
                                  verbose=int(args.verbose))
        clf.fit(X, y)

        train_acc = (clf.predict(X) == y).mean()
        test_acc = (clf.predict(Xt) == yt).mean()
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if args.verbose:
            print('lambda = {:10e}'.format(lmbda), end=' ', flush=True)
        print('train_acc: {:7.4f}'.format(train_acc), 'test_acc: {:7.4f}'.format(test_acc), end=' ')

        if lmbda_best is None or test_acc > acc_best:
            lmbda_best = lmbda
            acc_best = test_acc
            print('New best!')
        else:
            print()

    print('Best lambda:', lmbda_best, 'best test accuracy:', acc_best)

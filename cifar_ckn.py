import argparse
import json
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf

import algos
import solvers
import dataset
from ckn_encode_queue import CKNEncoder

import cifar10_input
import mnist_input
import stl10_input


cuda_device = 0
ENCODE_SIZE = 50000
CKN_BATCH_SIZE = 256


class DatasetIterator(object):
    def __init__(self, ds, layers):
        self.ds = ds
        self.layers = layers
        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.ds.init(self.sess, self.coord)
        self.encoder = None
        if self.ds.augmentation:
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.encoder = CKNEncoder([self.ds.images, self.ds.labels, self.ds.indexes],
                                      self.ds.test_features.shape[1],
                                      self.layers, batch_size=ENCODE_SIZE,
                                      cuda_device=cuda_device, ckn_batch_size=CKN_BATCH_SIZE)
            self.encoder.start_queue(self.sess, self.coord)

    def run(self, num_epochs):
        n = self.ds.train_data.shape[0]        
        for step in range(n * num_epochs // ENCODE_SIZE):
            epoch = float(step) * ENCODE_SIZE / n
            if self.ds.augmentation:
                X, labels, indexes = self.sess.run(
                        [self.encoder.encoded, self.encoder.labels, self.encoder.indexes])
                yield (epoch, X, labels, indexes)
            else:
                indexes = np.random.randint(n, size=ENCODE_SIZE)
                yield (epoch, None, None, indexes)

    def close(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.ds.close(self.sess, self.coord)
        self.encoder.join(self.sess, self.coord)
        self.sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Runs solvers with data-augmentation.")

    parser.add_argument('--experiment',
      default='cifar10',
      help='identifier of the experiment to run (dataset, model, etc)')
    parser.add_argument('--nepochs',
      default=10, type=int,
      help='number of epochs to run')
    parser.add_argument('--results-file',
      default='tmp.pkl',
      help='Filename of results pickle file.')
    parser.add_argument('--encoded-ds-filename',
      default=None,
      help='store encoded dataset in this npy file')
    parser.add_argument('--augmentation',
      default=True,
      action='store_true',
      help='whether to enable data augmentation')
    parser.add_argument('--no-augmentation',
      dest='augmentation',
      action='store_false',
      help='whether to disable data augmentation')
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
    parser.add_argument('--start-decay-step',
      default=20, type=int,
      help='step at which to start decaying the stepsize')
    parser.add_argument('--no-decay',
      default=False,
      action='store_true',
      help='whether to start decaying the stepsize at step specified at start-decay-step')
    parser.add_argument('--compute-loss',
      default=False,
      action='store_true',
      help='whether to compute train/test losses')

    args = parser.parse_args()
    print('experiment:', args.experiment, 'augmentation:', args.augmentation,
          'normalize:', args.normalize, 'no-decay:', args.no_decay)

    cuda_device = args.cuda_device

    # load experiment details
    if args.experiment == 'cifar10':
        model = cifar10_input.load_ckn_layers_whitened()
        data = cifar10_input.load_dataset_whitened()
        expt_params = cifar10_input.params()
        augm_fn = cifar10_input.augmentation
    elif args.experiment == 'mnist':
        model = mnist_input.load_ckn_layers()
        data = mnist_input.load_dataset()
        expt_params = mnist_input.params()
        augm_fn = mnist_input.augmentation
    elif args.experiment == 'infimnist':
        model = mnist_input.load_ckn_layers()
        data = mnist_input.load_dataset()
        expt_params = mnist_input.params()
        augm_fn = None  # no need for tf augmentation
    elif args.experiment.startswith('stl10'):
        # format: stl10_<fold><t[est]/v[al]>
        # e.g. stl10_3v for training with fold 3 and testing with
        # the rest of the training set as validation set
        fold = int(args.experiment[6])
        if args.experiment[7] == 't':
            data = stl10_input.load_train_test_white(fold)
        else:
            data = stl10_input.load_train_val_white(fold)
        model = stl10_input.load_ckn_layers_whitened()
        expt_params = stl10_input.params()
        augm_fn = stl10_input.augmentation
    else:
        print('experiment', args.experiment, 'not supported!')
        sys.exit(0)

    if 'ckn_batch_size' in expt_params:
        CKN_BATCH_SIZE = expt_params['ckn_batch_size']
    if 'encode_size' in expt_params:
        ENCODE_SIZE = expt_params['encode_size']

    with tf.device('/cpu:0'):
        if args.experiment == 'infimnist':
            ds = dataset.CKNInfimnistDataset(data, batch_size=ENCODE_SIZE, capacity=2*ENCODE_SIZE,
                                             cuda_device=cuda_device, ckn_batch_size=CKN_BATCH_SIZE)
        else:
            ds = dataset.CKNDataset(data, augmentation=args.augmentation, augm_fn=augm_fn,
                                    batch_size=ENCODE_SIZE, capacity=2*ENCODE_SIZE,
                                    cuda_device=cuda_device, ckn_batch_size=CKN_BATCH_SIZE)

    # leave here to avoid stream executor issues by creating session
    tf.Session()

    init_train = args.compute_loss or not ds.augmentation
    ds.init_test_features(model, init_train=init_train)

    n_classes = expt_params['n_classes']
    Xtest = ds.test_features.astype(solvers.dtype)
    if args.normalize:
        solvers.center(Xtest)
        solvers.normalize(Xtest)
    ytest = ds.test_labels
    if init_train:
        Xtrain = ds.train_features.astype(solvers.dtype)
        if args.normalize:
            solvers.center(Xtrain)
            solvers.normalize(Xtrain)
        ytrain = ds.train_labels

        if args.encoded_ds_filename:
            np.save(args.encoded_ds_filename,
                    [{'Xtr': Xtrain, 'ytr': ds.train_labels, 'Xte': Xtest, 'yte': ds.test_labels}])

    dim = Xtest.shape[1]
    n = ds.train_data.shape[0]
    print('n =', n, 'dim =', dim)

    engine = DatasetIterator(ds, model)

    loss = algos.LogisticLoss()
    lmbda = expt_params['lmbda']
    print('lmbda:', lmbda)

    solver_list = [
        solvers.MISOOneVsRest(n_classes, dim, n, lmbda=lmbda, loss=loss.name.encode('utf-8')),
    ]
    solver_params = [dict(name='miso_onevsrest', lmbda=lmbda, loss=loss.name)]
    # adjust miso step-size if needed (with L == 1)
    solver_list[0].decay(min(1, lmbda * n / (1 - lmbda)))

    lrs = expt_params['lrs']
    print('lrs:', lrs)
    for lr in lrs:
        solver_list.append(solvers.SGDOneVsRest(
                n_classes, dim, lr=lr, lmbda=lmbda, loss=loss.name.encode('utf-8')))
        solver_params.append(dict(name='sgd_onevsrest', lr=lr, lmbda=lmbda, loss=loss.name))

    n_algos = len(solver_list)
    start_time = time.time()

    test_accs = []
    train_losses = []
    test_losses = []
    epochs = []
    for step, (e, Xdata, labels, idxs) in enumerate(engine.run(args.nepochs)):
        t0 = time.time()
        if ds.augmentation:
            X = Xdata.astype(solvers.dtype)
            y = labels
            if args.normalize:
                solvers.center(X)
                solvers.normalize(X)
        else:
            X = Xtrain
            y = ytrain
        times = []
        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []
        for alg, solver in enumerate(solver_list):
            if not args.no_decay and step == args.start_decay_step:
                # print('starting stepsize decay')
                # solver.start_decay()
                print('decaying stepsize')
                solver.decay(0.5)

            t1 = time.time()
            if ds.augmentation:
                solver.iterate(X, y, idxs)
            else:
                solver.iterate_indexed(X, y, idxs)
            times.append(time.time() - t1)
            acc_train.append((solver.predict(X) == y).mean())
            acc_test.append((solver.predict(Xtest) == ytest).mean())
            if args.compute_loss:
                loss_train.append(solver.compute_loss(Xtrain, ytrain))
                loss_test.append(solver.compute_loss(Xtest, ytest))

        print('epoch', e)
        print('train batch acc', acc_train)
        print('test acc', acc_test)
        if args.compute_loss:
            print('train loss', loss_train)
            print('test loss', loss_test)
        t = time.time()
        print('elapsed time:', t - start_time,
              'training/evaluation elapsed time:', t - t0,
              'iterate times:', times)
        start_time = t
        epochs.append(e)
        test_accs.append(acc_test)
        if args.compute_loss:
            train_losses.append(loss_train)
            test_losses.append(loss_test)
        sys.stdout.flush()

        if step % 10 == 0:  # save stats every 10 steps
            pickle.dump({'params': solver_params, 'epochs': epochs,
                         'test_accs': test_accs, 'train_losses': train_losses,
                         'test_losses': test_losses},
                        open(os.path.join(expt_params.get('results_root', ''),
                                          args.results_file), 'wb'))

    engine.close()

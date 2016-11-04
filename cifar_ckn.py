import argparse
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf

import solvers
import dataset
from dataset_iterator import DatasetIterator

import cifar10_input
import mnist_input
import stl10_input

ENCODE_SIZE = 50000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Runs solvers with data-augmentation.")

    parser.add_argument('--experiment',
      default='cifar10',
      help='identifier of the experiment to run (dataset, model, etc)')
    parser.add_argument('--model-type',
      default='ckn',
      help='identifier of the model type (ckn, scattering)')
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
    parser.add_argument('--cuda-devices',
      default='0',
      help='CUDA GPU device numbers')
    parser.add_argument('--num-workers',
      default=10, type=int,
      help='number of processes for scattering encoding')
    parser.add_argument('--start-decay-epoch',
      default=10, type=float,
      help='epoch at which to start decaying the stepsize')
    parser.add_argument('--no-decay',
      default=False,
      action='store_true',
      help='whether to start decaying the stepsize at step specified at start-decay-step')
    parser.add_argument('--compute-loss',
      default=False,
      action='store_true',
      help='whether to compute train/test losses')

    args = parser.parse_args()
    print('experiment:', args.experiment, 'model-type:', args.model_type,
           'augmentation:', args.augmentation, 'normalize:', args.normalize,
           'no-decay:', args.no_decay)

    if not args.no_decay:
        print('decay starts in epoch', args.start_decay_epoch)

    model_params = {}
    if args.model_type == 'ckn':
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

        cuda_devices = list(map(int, args.cuda_devices.split()))
        assert len(cuda_devices) > 0, 'at least 1 GPU is needed for CKNs'
        model_params['cuda_devices'] = cuda_devices
        model_params['layers'] = model
        model_params['ckn_batch_size'] = expt_params.get('ckn_batch_size', 256)

    elif args.model_type == 'scattering':
        import scattering_encoder
        scattering_encoder.create_pool(args.num_workers)
        # load experiment details
        if args.experiment == 'cifar10':
            data = cifar10_input.load_dataset_raw()
            filters, m = cifar10_input.get_scattering_params()
            expt_params = cifar10_input.params_scat()
            augm_fn = cifar10_input.augmentation
        elif args.experiment.startswith('stl10'):
            # format: stl10_<fold><t[est]/v[al]>
            # e.g. stl10_3v for training with fold 3 and testing with
            # the rest of the training set as validation set
            fold = int(args.experiment[6])
            if args.experiment[7] == 't':
                data = stl10_input.load_train_test_raw(fold)
            else:
                data = stl10_input.load_train_val_raw(fold)
            def pad(im):
                padded = np.zeros((im.shape[0], 128, 128, im.shape[3]), dtype=np.float32)
                padded[:,16:112,16:112,:] = im
                return padded
            data = (pad(data[0]), data[1], pad(data[2]), data[3])
            filters, m = stl10_input.get_scattering_params()
            expt_params = stl10_input.params_scat()
            augm_fn = stl10_input.augmentation_scat
        else:
            print('experiment', args.experiment,
                  'not supported for model type', args.model_type)
            sys.exit(0)

        model_params['encoder'] = scattering_encoder.ScatteringEncoder(filters, m)

    else:
        print('model type', args.model_type, 'is not supported!')
        sys.exit(0)


    encode_size = expt_params.get('encode_size', ENCODE_SIZE)

    # data = tuple(data[i][:1000] for i in range(4))

    with tf.device('/cpu:0'):
        if args.experiment == 'infimnist':
            ds = dataset.InfimnistDataset(data, batch_size=encode_size, capacity=2*encode_size)
        else:
            ds = dataset.Dataset(data, augmentation=args.augmentation, augm_fn=augm_fn,
                                 batch_size=encode_size, capacity=2*encode_size)

    # leave here to avoid stream executor issues by creating session
    tf.Session()

    init_train = args.compute_loss or not ds.augmentation
    ds.init_features(args.model_type, model_params, init_train=init_train)

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
    start_decay_step = int(n * args.start_decay_epoch / encode_size)

    loss = b'squared_hinge'
    lmbdas = expt_params['lmbda']
    if not isinstance(lmbdas, list):
        lmbdas = [lmbdas]

    for lmbda in lmbdas:
        print('\n\nLambda:', lmbda)

        solver_list = []
        solver_params = []
        miso_lrs = expt_params.get('miso_lrs', [1])
        print('miso lrs:', miso_lrs)
        for lr in miso_lrs:
            solver_list.append(solvers.MISOOneVsRest(n_classes, dim, n, lmbda=lmbda, loss=loss))
            solver_params.append(dict(name='miso_onevsrest', lmbda=lmbda, loss=loss, lr=lr))
            # adjust miso step-size if needed (with L == 1)
            if loss in [b'logistic', b'squared_hinge']:
                solver_list[-1].decay(lr * min(1, lmbda * n))

        saga_lrs = expt_params.get('saga_lrs', [1])
        print('saga lrs:', saga_lrs)
        for lr in saga_lrs:
            solver_list.append(solvers.SAGAOneVsRest(n_classes, dim, n,
                                                     lr=lr, lmbda=lmbda, loss=loss))
            solver_params.append(dict(name='saga_onevsrest', lmbda=lmbda, loss=loss, lr=lr))

        lrs = expt_params['lrs']
        print('lrs:', lrs)
        for lr in lrs:
            solver_list.append(solvers.SGDOneVsRest(
                    n_classes, dim, lr=lr, lmbda=lmbda, loss=loss))
            solver_params.append(dict(name='sgd_onevsrest', lr=lr, lmbda=lmbda, loss=loss))

        start_time = time.time()

        engine = DatasetIterator(ds, args.model_type, model_params, encode_size=encode_size)

        test_accs = []
        train_losses = []
        regs = []
        test_losses = []
        epochs = []

        def save():
            pickle.dump({'params': solver_params, 'epochs': epochs,
                         'test_accs': test_accs, 'train_losses': train_losses,
                         'test_losses': test_losses, 'regs': regs},
                        open(os.path.join(
                         expt_params.get('results_root', ''),
                         args.model_type + '_' + args.results_file.format(lmbda=lmbda)), 'wb'))

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
            reg = []
            loss_train = []
            loss_test = []
            for alg, solver in enumerate(solver_list):
                if not args.no_decay and step == start_decay_step:
                    print('starting stepsize decay')
                    solver.start_decay()
                    # print('halving stepsize')
                    # solver.decay(0.5)

                t1 = time.time()
                if ds.augmentation:
                    solver.iterate(X, y, idxs)
                else:
                    solver.iterate_indexed(X, y, idxs)
                times.append(time.time() - t1)
                acc_train.append((solver.predict(X) == y).mean())
                acc_test.append((solver.predict(Xtest) == ytest).mean())
                if args.compute_loss:
                    reg.append(0.5 * lmbda * solver.compute_squared_norm())
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
                regs.append(reg)
                train_losses.append(loss_train)
                test_losses.append(loss_test)
            sys.stdout.flush()

            if step % 2 == 0:  # save stats every 2 steps
                save()

        save()
        engine.close()

    if args.model_type == 'scattering':
        scattering_encoder.close_pool()

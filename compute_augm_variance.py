import argparse
import os
import pickle
import sys
import time
import numpy as np
import tensorflow as tf

import dataset

import cifar10_input
import mnist_input
import stl10_input


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description="Helper for computing variance of transformations")

    parser.add_argument('--experiment',
      default='cifar10',
      help='identifier of the experiment to run (dataset, model, etc)')
    parser.add_argument('--model-type',
      default='ckn',
      help='identifier of the model type (ckn, scattering)')
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
    parser.add_argument('--num-examples',
      default=100, type=int,
      help='number of training examples to consider for computing variance')
    parser.add_argument('--num-transformations',
      default=100, type=int,
      help='number of transformations to consider per example')

    args = parser.parse_args()
    print('experiment:', args.experiment, 'model-type:', args.model_type,
          'normalize:', args.normalize)

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

            def resize(im, sz=64):
                from scipy.misc import imresize
                resized = np.zeros((im.shape[0], sz, sz, im.shape[3]), dtype=np.float32)
                for i in range(im.shape[0]):
                    resized[i] = imresize(im[i], (sz, sz, im.shape[3])).astype(np.float32) / 255
                return resized

            data = (resize(data[0]), data[1], resize(data[2]), data[3])
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

    # data augmentation
    if args.experiment == 'infimnist':
        idxs = np.hstack([10000 + i * 60000 + np.arange(args.num_examples)
                          for i in range(args.num_transformations)])
        from infimnist._infimnist import InfimnistGenerator
        images, _ = InfimnistGenerator().gen(idxs)
        idxs = (idxs - 10000) % 60000
        images = images.astype(np.float32).reshape(-1, 1, 28, 28) / 255
    else:
        data = tuple(data[i][:args.num_examples] for i in range(4))
        num_images = args.num_examples * args.num_transformations
        with tf.device('/cpu:0'):
            ds = dataset.Dataset(data, augmentation=True, augm_fn=augm_fn,
                                 batch_size=num_images, capacity=num_images,
                                 producer_type='epoch')
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            ds.init(sess, coord)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            idxs, images = sess.run([ds.indexes, ds.images])

            coord.request_stop()
            coord.join(threads)
            ds.close(sess, coord)

    print('encoding...')
    t = time.time()
    if args.model_type == 'ckn':
        import _ckn_cuda as ckn
        cuda_device = model_params['cuda_devices'][0]
        N, C, H, W = images.shape
        Xfull = ckn.encode_cudnn(np.ascontiguousarray(images.reshape(N, C*H, W)), model_params['layers'],
                             cuda_device, model_params['ckn_batch_size'])
    elif args.model_type == 'scattering':
        Xfull = model_params['encoder'].encode_nchw(images)
        scattering_encoder.close_pool()
    print('donc encoding, elapsed:', time.time() - t)

    print('computing variances...')
    # all data
    full_var = ((Xfull - Xfull.mean(0)) ** 2).sum(1).mean()

    # per example
    vars = []
    for idx in range(args.num_examples):
        X = Xfull[idxs == idx, ...]
        vars.append(((X - X.mean(0)) ** 2).sum(1).mean())

    print('full var:', full_var)
    print('mean idx var:', np.mean(vars), 'median idx var', np.median(vars), 'max idx var', np.max(vars))

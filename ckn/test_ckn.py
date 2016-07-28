#! /usr/bin/env python3

import sys
import os
import shutil
import argparse
import json

import numpy as np
import scipy
import pickle
from enum import IntEnum

import _ckn_cuda as ckn

def unpickle_cifar(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

class LayerType(IntEnum):
  raw_patchmap = 0
  centering = 1
  center_whiten = 2
  center_shift = 3
  gradient_2d = 4
  center_local_whiten = 5

def read_dataset_cifar10(folder):
  """read the pickle files provided from 
  https://www.cs.toronto.edu/~kriz/cifar.html
  and returns all data in one numpy array for training and one for testing"""

  n_batch = 10000
  n_train = 5*n_batch
  n_test = n_batch

  Xtr = np.empty((n_train, 96, 32), dtype=np.float32)
  Ytr = np.empty(n_train, dtype=np.float32)
  for i in range(1, 6):
    d = unpickle_cifar(os.path.join(folder, 'data_batch_{}'.format(i)))
    Xtr[(i-1)*n_batch:i*n_batch] = d[b'data'].reshape(n_batch, 96, 32)/255.0
    Ytr[(i-1)*n_batch:i*n_batch] = d[b'labels']

  d = unpickle_cifar(os.path.join(folder, 'test_batch'))
  Xte = np.ascontiguousarray(d[b'data'].astype(np.float32).reshape(n_batch, 96, 32))/255.0
  Yte = np.array(d[b'labels'], dtype=np.float32)
  return Xtr, Ytr, Xte, Yte


# FIXME: does not handle zero padding
def extract_random_3D_patches_as_vectors(patch_shape, involume, patches_per_input):
    """
    extracts 3D patches as column vectors and puts them into one big matrix
    """
    involume = np.ascontiguousarray(involume)  # won't make a copy if not needed
    N, T, Y, X, channels = involume.shape
    count = np.sum(patches_per_input)
    patchmap_shape = (T - patch_shape[0] + 1, Y - patch_shape[1] + 1, X - patch_shape[2] + 1,
                      patch_shape[0], patch_shape[1], patch_shape[2],
                      channels)

    result = np.empty((count, np.prod(patchmap_shape[3:])), dtype=involume.dtype)
    idx_patch = 0
    for n in range(N):
      for i in range(patches_per_input[n]):
        for attempt in range(10):
          t = np.random.randint(patchmap_shape[0])
          y = np.random.randint(patchmap_shape[1])
          x = np.random.randint(patchmap_shape[2])
          patch = involume[n, t:t+patch_shape[0], y:y+patch_shape[1], x:x+patch_shape[2]]
          if np.linalg.norm(patch) >= 10e-6:
            break
        result[idx_patch] = patch.flatten()
        idx_patch += 1
    assert idx_patch == count, (idx_patch, count)

    return result


def make_training_set(inputs, layers, npatch, cuda_device, n_patch=1000000):
  N = inputs.shape[0]
  batch_size = 4096
  patches_per_input, rest = divmod(n_patch, N)
  print('patches_per_input: {}'.format(patches_per_input))
  patches_per_input = np.full(shape=(N,), fill_value=patches_per_input, dtype=np.int32)
  patches_per_input[:rest] += 1
  patch_start_idxs = np.cumsum(patches_per_input)
  patch_start_idxs = [0] + list(patch_start_idxs)

  for i in range(0, N, batch_size):
    end_idx = min(N, i+batch_size)

    if layers:
      psi_i = ckn.encode_cudnn(inputs[i:end_idx], layers, cuda_device, 64, verbose=1)
      
      sqrtshp = np.sqrt(psi_i.shape[1]/layers[-1]['nfilters']).astype(np.int)
      psi_i = psi_i.reshape(end_idx-i, sqrtshp, sqrtshp, layers[-1]['nfilters'])
    else:
      psi_i = inputs[i:end_idx].reshape(end_idx-i, 3, 32, 32).transpose(0, 2, 3, 1)

    psi_i.shape = (psi_i.shape[0], 1) + psi_i.shape[1:]
    print(psi_i.shape)
    X_i = extract_random_3D_patches_as_vectors(
        (1, npatch, npatch),
        psi_i,
        patches_per_input[i:end_idx])

    if i == 0:
      X = np.empty((n_patch, X_i.shape[1]), dtype=psi_i.dtype)
    X[patch_start_idxs[i]:patch_start_idxs[end_idx]] = X_i
    del psi_i
    print(' {}'.format(end_idx), end='\r', flush=True)
  print()
  np.random.shuffle(X)
  norms = np.linalg.norm(X, axis=1, keepdims=True)
  X /= norms
  X = np.ascontiguousarray(X[norms.squeeze() >= 10e-6])
  return X


def train_ckn(inputs, layers, outfolder, cuda_device):
  for idx_layer in range(len(layers)):
    layer_fname = os.path.join(outfolder, 'layers_{}.npy'.format(idx_layer))
    if os.path.isfile(layer_fname):
      print('found file for layer {}: {}'.format(idx_layer, layer_fname))
      layers = np.load(layer_fname)
    else:
      print('training layer {}'.format(idx_layer))
      layer = layers[idx_layer]

      X = make_training_set(inputs, layers[:idx_layer], layer['npatch'], cuda_device, 100000)

      if 'sigma' not in layer:
        print('Given quantile {}, compute sigma... '
            .format(layer['sigma_quantile']), end='', flush=True)
        layer['sigma'] = ckn.compute_sigma(X, layer['sigma_quantile'])
        print(layer['sigma'])
      else:
        print('sigma: {}'.format(layer['sigma']))

      print('training {} layer'.format(layer['type_layer']))
      if layer['type_layer'] == 'multi_projection':
        layer['subspace_centroids'], layer['W'], layer['b'], layer['W2'], layer['W3'] = ckn.train_layer_multiprojection(
            X,
            layer['sigma'],
            layer['lambda2'],
            layer['nfilters'],
            layer['num_subspaces'],
            iter_kmeans=10)
      else:
        layer['W'], layer['b'], layer['W2'], layer['W3'] = ckn.train_layer(
            X,
            layer['sigma'],
            layer['lambda2'],
            layer['nfilters'],
            iter_kmeans=10)
      np.save(layer_fname, layers)

  return layers


def get_gpu_device():
  gpu_devices = [0]
  try:
    gpu_devices = os.environ['CUDA_VISIBLE_DEVICES'].split()
  except KeyError:
    print('''

      WARNING: CUDA_VISIBLE_DEVICES not in environment, defaulting to GPU 0

      ''')
    assert len(gpu_devices) == 1, 'cannot use multiple gpus'
  return int(gpu_devices[0])


if __name__=="__main__":

  parser = argparse.ArgumentParser(
      description="Trains a network and encodes the dataset with it.")

  parser.add_argument('--network',
      default='network.json',
      help='JSON file holding network hyperparameters')
  parser.add_argument('--dataset-folder',
      default="/home/clear/dwynen/ckn-dataset/cifar-10-batches-py/",
      help="Folder containing dataset files.")
  parser.add_argument('--results-root',
      default='/scratch/clear/abietti/results/ckn',
      help='Root folder for results. Will make a subfolder there based on $tag')
  parser.add_argument('--tag',
      required=True,
      help='name of the subfolder for experiment to make')
  parser.add_argument('-v', '--verbose', action='count',
      help="verbosity level")

  args = parser.parse_args()
  

  outfolder = os.path.join(args.results_root, args.tag)
  os.makedirs(outfolder, exist_ok=True)

  print('\nloading dataset batches from {}'.format(args.dataset_folder))
  Xtr, Ytr, Xte, Yte = read_dataset_cifar10(args.dataset_folder)
  print('dataset loaded')

  cuda_device = get_gpu_device()
  cuda_device = 0
  print('\nusing GPU {}'.format(cuda_device))

  print('\nusing network architecture from {}'.format(args.network))
  network_file = os.path.join(outfolder, 'network.json')
  print('making copy of json in {}'.format(network_file))
  shutil.copyfile(args.network, network_file)
  with open(network_file) as f:
    layers = json.load(f)


  print('\nbeginning training')
  layers = train_ckn(Xtr, layers, outfolder, cuda_device)

  # print('encoding train...')
  # psiTr = ckn.encode_cudnn(Xtr, layers, cuda_device, 128, verbose=1)
  # np.savez(os.path.join(outfolder, 'fmaps_tr.npz'), X=psiTr, Y=Ytr)

  # print('encoding test...')
  # psiTe = ckn.encode_cudnn(Xte, layers, cuda_device, 128, verbose=1)
  # np.savez(os.path.join(outfolder, 'fmaps_te.npz'), X=psiTe, Y=Yte)
  



__doc__ = '''Algorithms for training the ERM objective.'''

import numpy as np


class Loss(object):
  name = None

  def compute(self, preds, y):
    raise NotImplementedError()

  def compute_gradient(self, preds, y):
    raise NotImplementedError()


class SquaredLoss(Loss):
  name = 'l2'

  def compute(self, preds, y):
    return 0.5 * (preds - y) ** 2

  def compute_gradient(self, preds, y):
    return (preds - y)[:,None] * X


class LogisticLoss(Loss):
  name = 'logistic'

  def compute(self, preds, y):
    sigm = 1. / (1 + np.exp(-preds))
    return - y * np.log(sigm) - (1 - y) * np.log(1 - sigm)

  def compute_gradient(self, preds, y):
    sigm = 1. / (1 + np.exp(-preds))
    return sigm - y


class Solver(object):
  def iterate(self, x, y, idx):
    raise NotImplementedError()


class SGD(Solver):
  def __init__(self, w0, lr=0.1, decay=1.0, lmbda=0., loss=None):
    self.w = w0.copy()
    self.lr = lr
    self.decay = decay
    self.lmbda = lmbda  # regularization param
    self.loss = loss or LogisticLoss()

  def begin_epoch(self, epoch):
    if epoch > 0 and self.decay < 1.0:
      self.lr *= self.decay

  def iterate(self, x, y, idx):
    self.w = self.w - self.lr * (
        self.loss.compute_gradient(x.dot(self.w), y) + self.lmbda * self.w)


class SVRG(Solver):
  def __init__(self, w0, lr=0.1, lmbda=0., loss=None):
    self.w = w0.copy()
    self.w_tilde = None
    self.grad_tilde = None
    self.lr = lr
    self.lmbda = lmbda  # regularization param
    self.loss = loss or LogisticLoss()

  def begin_pass(self, X, y):
    '''Compute "full" gradient at the start of each pass.'''
    # TODO: don't require to have all of X in memory
    self.w_tilde = self.w
    self.grad_tilde = self.loss.compute_gradient(X.dot(self.w_tilde), y).mean(0)

  def iterate(self, x, y, idx):
    self.w = self.w - self.lr * (
        self.loss.compute_gradient(x.dot(self.w), y)
        - self.loss.compute_gradient(x.dot(self.w_tilde), y)
        + self.grad_tilde + self.lmbda * self.w)


class MISO(Solver):
  def __init__(self, w0, n, lmbda=0.1, loss=None):
    self.w = w0.copy()
    self.n = n  # number of examples in training set
    self.lmbda = lmbda  # regularization param (= 1 / stepsize)
    self.loss = loss or LogisticLoss()

    self.z = np.zeros((self.n, self.w.shape[0]))

  def begin_epoch(self, epoch):
    pass

  def iterate(self, x, y, idx):
    # zi <- w - 1/lambda (grad f_i + lambda w)
    zi = - 1. / self.lmbda * self.loss.compute_gradient(x.dot(self.w), y)
    self.w = self.w + 1. / self.n * (zi - self.z[idx])
    self.z[idx] = zi

__doc__ = '''Algorithms for training the ERM objective.'''

import numpy as np


class Loss(object):
  name = None

  def compute(self, preds, y):
    raise NotImplementedError()

  def compute_gradient(self, X, preds, y):
    raise NotImplementedError()


class SquaredLoss(Loss):
  name = 'l2'

  def compute(self, preds, y):
    return 0.5 * (preds - y) ** 2

  def compute_gradient(self, X, preds, y):
    err = preds - y
    if len(X.shape) > 1:
      err = err[:,None]

    return err * X


class LogisticLoss(Loss):
  name = 'logistic'

  def __init__(self, grad_sigma=None):
    self.grad_sigma = grad_sigma

  def compute(self, preds, y):
    sigm = 1. / (1 + np.exp(-preds))
    return - y * np.log(sigm) - (1 - y) * np.log(1 - sigm)

  def compute_gradient(self, X, preds, y):
    sigm = 1. / (1 + np.exp(-preds))
    err = sigm - y
    if len(X.shape) > 1:
      err = err[:,None]
    return err * X + (0. if self.grad_sigma is None else self.grad_sigma * np.random.randn(*err.shape))


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

    self.wavg = self.w
    self.t = 1

  def begin_epoch(self, epoch):
    if epoch > 0 and self.decay < 1.0:
      self.lr *= self.decay

  def iterate(self, x, y, idx):
    self.w = self.w - self.lr / self.t * (
        self.loss.compute_gradient(x, x.dot(self.w), y) + self.lmbda * self.w)
    self.t += 1
    self.wavg = (1 - 1./self.t) * self.wavg + 1./self.t * self.w


class SVRG(Solver):
  def __init__(self, w0, lr=0.1, lmbda=0., loss=None, average=False):
    self.w = w0.copy()
    self.w_tilde = None
    self.grad_tilde = None
    self.lr = lr
    self.lmbda = lmbda  # regularization param
    self.loss = loss or LogisticLoss()
    self.average = average  # whether to average iterates after each pass

    self.t = 1
    self.wavg = self.w

  def begin_pass(self, X, y):
    '''Compute "full" gradient at the start of each pass.'''
    # TODO: don't require to have all of X in memory
    self.w_tilde = self.wavg if self.average else self.w
    self.grad_tilde = self.loss.compute_gradient(X, X.dot(self.w_tilde), y).mean(0)

    self.w = self.w_tilde
    self.t = 1
    self.wavg = self.w

  def iterate(self, x, y, idx):
    self.w = self.w - self.lr * (
        self.loss.compute_gradient(x, x.dot(self.w), y)
        - self.loss.compute_gradient(x, x.dot(self.w_tilde), y)
        + self.grad_tilde + self.lmbda * self.w)
    self.t += 1
    self.wavg = (1 - 1./self.t) * self.wavg + 1./self.t * self.w


class MISO(Solver):
  def __init__(self, w0, n, lmbda=0.1, decay=False, loss=None, compute_lb=False):
    self.w = w0.copy()
    self.n = n  # number of examples in training set
    self.lmbda = lmbda  # regularization param (= 1 / stepsize)
    self.loss = loss or LogisticLoss()
    self.decay = decay
    self.compute_lb = compute_lb  # to compute lower bounds

    self.z = np.zeros((self.n, self.w.shape[0]))
    self.c = None
    if self.compute_lb:
      self.c = np.zeros(self.n)

    self.t = 1
    self.wavg = self.w

  def begin_epoch(self, epoch):
    pass

  def iterate(self, x, y, idx):
    pred = x.dot(self.w)
    g = self.loss.compute_gradient(x, pred, y)
    if len(g.shape) > 1:
        g = g.mean(0)
    step = 1.
    if self.decay:
        step = min(1., 2. * self.n / (self.t + 1))
    # zi <- w - 1/lambda (grad f_i + lambda w)
    zi = (1 - step) * self.z[idx] - step / self.lmbda * g
    if self.compute_lb:
      self.c[idx] = float(1 - step) * self.c[idx] + float(step) * (
              self.loss.compute(pred, y) - self.w.dot(g))

    self.w = self.w + 1. / self.n * (zi - self.z[idx])
    self.z[idx] = zi

    self.t += 1
    self.wavg = (1 - 1./self.t) * self.wavg + 1./self.t * self.w

  def lower_bound(self, w):
    assert self.compute_lb, "compute_lb wasn't set to True"
    return (self.c - self.lmbda * self.z.dot(w)).mean() + 0.5 * self.lmbda * (w ** 2).sum()

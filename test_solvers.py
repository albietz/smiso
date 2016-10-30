import logging
import numpy as np
import solvers
import time

from sklearn import datasets


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def load_dataset(digit=4, sparse=False):
    digits = datasets.load_digits()
    X = (digits.data / 16.).astype(solvers.dtype)
    if sparse:
        import scipy.sparse as sp
        X = sp.csr_matrix((X > 0.6).astype(solvers.dtype))
    if digit is None:
        y = digits.target.astype(np.int32)
    else:
        y = (digits.target == digit).astype(solvers.dtype)
    return X, y


def test_sgd():
    X, y = load_dataset()
    solver = solvers.SGD(X[0].shape[0])
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    assert not np.any(np.isnan(solver.w))


def test_sgd_onevsrest():
    loss = b'squared_hinge'
    X, y = load_dataset(digit=None)
    solver = solvers.SGDOneVsRest(y.max() + 1, X[0].shape[0], loss=loss)
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))


def test_sparse_sgd():
    X, y = load_dataset(sparse=True)
    solver = solvers.SparseSGD(X.shape[1], loss=b'squared_hinge')
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    assert not np.any(np.isnan(solver.w))
    logger.info('sparse sgd loss: %f', solver.compute_loss(X, y))

    Xd = X.todense()
    dsolver = solvers.SGD(Xd.shape[1], loss=b'squared_hinge')
    dsolver.iterate(Xd, y, np.arange(Xd.shape[0], dtype=np.int64))
    logger.info('dense sgd loss: %f', dsolver.compute_loss(Xd, y))
    assert np.sum(np.square(solver.w - dsolver.w)) < 1e-8, \
            'sparse and dense SGD do not match!'


def test_miso():
    X, y = load_dataset()
    solver = solvers.MISO(X[0].shape[0], X.shape[0])
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    assert not np.any(np.isnan(solver.w))


def test_sparse_miso():
    X, y = load_dataset(sparse=True)
    solver = solvers.SparseMISO(X.shape[1], X.shape[0], loss=b'squared_hinge')
    solver.init(X)
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    assert not np.any(np.isnan(solver.w))
    logger.info('sparse miso loss: %f', solver.compute_loss(X, y))

    Xd = X.todense()
    dsolver = solvers.MISO(Xd.shape[1], X.shape[0], loss=b'squared_hinge')
    dsolver.iterate(Xd, y, np.arange(Xd.shape[0], dtype=np.int64))
    logger.info('dense miso loss: %f', dsolver.compute_loss(Xd, y))
    assert np.sum(np.square(solver.w - dsolver.w)) < 1e-8, \
            'sparse and dense MISO do not match!'


def test_saga():
    X, y = load_dataset()
    solver = solvers.SAGA(X[0].shape[0], X.shape[0])
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    assert not np.any(np.isnan(solver.w))


def test_sparse_saga():
    X, y = load_dataset(sparse=True)
    lmbda = 0.1
    solver = solvers.SparseSAGA(X.shape[1], X.shape[0], lr=0.1, lmbda=lmbda, loss=b'squared_hinge')
    solver.init(X)
    solver.iterate(X, y, np.arange(X.shape[0], dtype=np.int64))
    # solver.iterate(X[:2], y[:2], np.arange(2, dtype=np.int64))
    assert not np.any(np.isnan(solver.w))
    logger.info('sparse saga loss: %f', solver.compute_loss(X, y) + 0.5 * lmbda * solver.compute_squared_norm())

    Xd = X.todense()
    dsolver = solvers.SAGA(Xd.shape[1], X.shape[0], lr=0.1, lmbda=lmbda, loss=b'squared_hinge')
    dsolver.iterate(Xd, y, np.arange(Xd.shape[0], dtype=np.int64))
    # dsolver.iterate(Xd[:2], y[:2], np.arange(2, dtype=np.int64))
    logger.info('dense saga loss: %f', dsolver.compute_loss(Xd, y) + 0.5 * lmbda * dsolver.compute_squared_norm())
    assert np.sum(np.square(solver.w - dsolver.w)) < 1., \
            'sparse and dense SAGA do not match!'


def benchmark():
    X, y = load_dataset(digit=None)
    sgd_solver = solvers.SGDOneVsRest(y.max() + 1, X.shape[1])
    miso_solver = solvers.MISOOneVsRest(y.max() + 1, X.shape[1], X.shape[0])
    t_max = 50

    idxs = np.random.randint(X.shape[0], size=X.shape[0])
    idxs2 = np.random.randint(X.shape[0], size=10*X.shape[0])

    t0 = time.time()
    for t in range(t_max):
        sgd_solver.iterate(X, y, idxs)
    print('sgd iterate:', time.time() - t0)
    t0 = time.time()
    for t in range(t_max):
        sgd_solver.iterate_indexed(X, y, idxs2)
    print('sgd iterate_indexed:', time.time() - t0)
    t0 = time.time()
    for t in range(t_max):
        miso_solver.iterate(X, y, idxs)
    print('miso iterate:', time.time() - t0)
    t0 = time.time()
    for t in range(t_max):
        miso_solver.iterate_indexed(X, y, idxs2)
    print('miso iterate_indexed:', time.time() - t0)


if __name__ == '__main__':
    test_sgd()
    test_sgd_onevsrest()
    test_sparse_sgd()
    test_miso()
    test_sparse_miso()
    test_saga()
    test_sparse_saga()

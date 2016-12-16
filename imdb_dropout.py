import argparse
import logging
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import solvers

from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

DATA_FOLDER = '/scratch/clear/abietti/data/aclImdb'

params = {
    'lmbda': [1e-2],
    'delta': [0.01, 0.1, 0.3],
    # 'delta': [0],
    'algos': [
        {'name': 'miso_nonu', 'lr': 1.0},
        {'name': 'miso', 'lr': 10.0},
        {'name': 'sgd', 'lr': 10.0},
        {'name': 'saga', 'lr': 10.0},
    ],
}

start_decay = 2
num_epochs = 400
loss = b'squared_hinge'
eval_delta = 5

def load_imdb():
    def process_imdb(X, y):
        return X.astype(solvers.dtype), (y > 5).astype(solvers.dtype)
    #     return sp.vstack((X[y <= 4], X[y >= 7])).astype(solvers.dtype), \
    #               np.hstack((np.zeros(np.sum(y <= 4)), np.ones(np.sum(y >= 7)))).astype(solvers.dtype)

    from sklearn.datasets import load_svmlight_files
    Xtrain, ytrain, Xtest, ytest = load_svmlight_files(
        (os.path.join(DATA_FOLDER, 'train/labeledBow.feat'),
         os.path.join(DATA_FOLDER, 'test/labeledBow.feat')))
    Xtrain, ytrain = process_imdb(Xtrain, ytrain)
    Xtest, ytest = process_imdb(Xtest, ytest)
    return Xtrain, ytrain, Xtest, ytest


def training(lmbda, dropout_rate, solver, q=None):
    ep = []
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    t_start = time.time()
    for epoch in range(num_epochs):
        if epoch % eval_delta == 0:
            ep.append(epoch)
            loss_train.append(solver.compute_loss(Xtrain, ytrain) + 0.5 * lmbda * solver.compute_squared_norm())
            loss_test.append(solver.compute_loss(Xtest, ytest))
            acc_train.append((((2*ytrain - 1) * Xtrain.dot(solver.w)) >= 0).mean())
            acc_test.append((((2*ytest - 1) * Xtest.dot(solver.w)) >= 0).mean())

        if epoch == start_decay:
            solver.start_decay()

        if dropout_rate > 0:
            idxs = np.random.choice(n, n, p=q)
            Xtt = Xtrain[idxs]
            Xtt.sort_indices()
            Xtt.data *= np.random.binomial(1, 1 - dropout_rate, size=Xtt.nnz) / (1 - dropout_rate)
    #         Xtt *= np.random.binomial(1, 1 - dropout_rate, size=Xtt.shape) / (1 - dropout_rate)
            t = time.time()
            solver.iterate(Xtt, ytrain[idxs], idxs)
        else:
            idxs = np.random.choice(n, n, p=q)
            t = time.time()
            solver.iterate_indexed(Xtrain, ytrain, idxs)

    logging.info('elapsed: %f', time.time() - t_start)
    # print('lmbda', lmbda, 'delta', dropout_rate,
    #       '=> train loss:', loss_train[-1], 'test loss:', loss_test[-1],
    #       'train acc:', acc_train[-1], 'test acc:', acc_test[-1])
    return {
        'epochs': ep,
        'loss_train': loss_train,
        'loss_test': loss_test,
        'acc_train': acc_train,
        'acc_test': acc_test,
    }


def train_sgd(lmbda, dropout_rate, lr):
    solver = solvers.SparseSGD(d, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss)
    return training(lmbda, dropout_rate, solver)

def train_miso(lmbda, dropout_rate, lr):
    solver = solvers.SparseMISO(d, n, lmbda=lmbda, loss=loss)
    solver.init(Xtrain, init_q=False)
    solver.decay(lr * min(1, n * lmbda * (1 - dropout_rate)**2 / Lmax))
    return training(lmbda, dropout_rate, solver)

def train_miso_nonu(lmbda, dropout_rate, lr):
    solver = solvers.SparseMISO(d, n, lmbda=lmbda, loss=loss)
    solver.init(Xtrain, init_q=False)
    q = np.asarray(Xtrain.power(2).sum(1)).flatten()
    q += q.mean()
    q /= q.sum()
    solver.set_q(q)
    solver.decay(lr * min(1, n * lmbda * (1 - dropout_rate)**2 / Lavg))
    return training(lmbda, dropout_rate, solver, q=q)

def train_saga(lmbda, dropout_rate, lr):
    solver = solvers.SparseSAGA(d, n, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss)
    solver.init(Xtrain)
    return training(lmbda, dropout_rate, solver)

train_fn = {'sgd': train_sgd, 'miso': train_miso, 'miso_nonu': train_miso_nonu, 'saga': train_saga}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dropout training on imdb')
    parser.add_argument('--num-workers', default=1,
                        type=int, help='number of threads for grid search')
    parser.add_argument('--pdf-file', default=None, help='pdf file to save to')
    args = parser.parse_args()

    logging.info('loading imdb data')
    Xtrain, ytrain, Xtest, ytest = load_imdb()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]

    Lmax = Xtrain.power(2).sum(1).max()
    Lavg = Xtrain.power(2).sum(1).mean()

    pp = None
    if args.pdf_file:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        import curves
        pp = PdfPages(args.pdf_file)

    logging.info('training')
    futures = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for lmbda in params['lmbda']:
            for delta in params['delta']:
                futures.append(((lmbda, delta),
                    [(alg, executor.submit(train_fn[alg['name']], lmbda, delta, alg['lr']))
                     for alg in params['algos']]))

        for (lmbda, delta), futs in futures:
            res = [(alg, f.result()) for alg, f in futs]
            print('lmbda', lmbda, 'delta', delta)
            for alg, r in res:
                print(alg['name'], alg['lr'], 'train loss', r['loss_train'][-1],
                      'test acc', r['acc_test'][-1])
            if pp:
                plot_res = {}
                plot_res['params'] = [dict(name=alg['name'], lr=alg['lr'], lmbda=lmbda, loss=loss)
                                      for alg, r in res]
                plot_res['epochs'] = res[0][1]['epochs']
                def transpose(key):
                    return list(zip(*(r[key] for alg, r in res)))
                plot_res['test_accs'] = transpose('acc_test')
                plot_res['train_accs'] = transpose('acc_train')
                plot_res['train_losses'] = transpose('loss_train')
                plot_res['test_losses'] = transpose('loss_test')

                curves.plot_loss(plot_res, ty='train', log=True, step=1, last=None,
                                 small=False, legend=True, title='imdb, $\delta$ = {:.2f}'.format(delta))
                pp.savefig()

    if pp:
        pp.close()

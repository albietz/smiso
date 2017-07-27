import argparse
import logging
import os
import sys
import time

import numpy as np
import solvers

from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

avg_experiment = False

if not avg_experiment:
    params = {
        'lmbda': [0.003],
        'delta': [0, 0.01, 0.1, 0.3, 0.5],
        'algos': [
            {'name': 'miso', 'lr': 0.1},
            {'name': 'miso', 'lr': 1.0},
            {'name': 'sgd', 'lr': 0.1},
            {'name': 'sgd', 'lr': 1.0},
            {'name': 'saga', 'lr': 0.1},
            {'name': 'saga', 'lr': 1.0},
            {'name': 'svrg', 'lr': 0.1},
            {'name': 'svrg', 'lr': 1.0},
        ],
    }

    start_decay = 2
    num_epochs = 501
else:
    params = {
        # 'lmbda': [0.003],
        'lmbda': [0.0003],
        # 'delta': [0, 0.01, 0.1, 0.3, 0.5],
        'delta': [0.1],
        'algos': [
            {'name': 'miso', 'lr': 0.1},
            {'name': 'miso_avg', 'lr': 0.1},
            {'name': 'sgd', 'lr': 0.1},
            {'name': 'sgd_avg', 'lr': 0.1},
        ],
    }

    # start_decay = 3
    start_decay = 30
    # num_epochs = 1001
    num_epochs = 1501


loss = b'logistic'
eval_delta = 5
eval_mc_samples = 0
seed = None

def load_breast_cancer():
    from scipy.io import loadmat
    mat = loadmat('/scratch/clear/abietti/data/vant.mat')
    X = np.ascontiguousarray(mat['X'].astype(solvers.dtype))
    y = (mat['Y'][:,1] == 1).astype(solvers.dtype)

    solvers.center(X)
    solvers.normalize(X)
    print(X.shape)
    train_size = 200
    Xtrain, Xtest = X[:train_size], X[train_size:]
    ytrain, ytest = y[:train_size], y[train_size:]
    return Xtrain, ytrain, Xtest, ytest


def training(lmbda, dropout_rate, solver, q=None):
    ep = []
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    random = np.random.RandomState(seed=seed or 43)
    # prepare evaluation set
    if dropout_rate > 0 and eval_mc_samples > 0:
        Xtrain_eval = np.vstack((Xtrain for _ in range(eval_mc_samples)))
        Xtrain_eval *= random.binomial(1, 1 - dropout_rate, size=Xtrain_eval.shape) / (1 - dropout_rate)
        ytrain_eval = np.hstack((ytrain for _ in range(eval_mc_samples)))
    else:
        Xtrain_eval = Xtrain
        ytrain_eval = ytrain

    t_start = time.time()
    for epoch in range(num_epochs):
        if epoch % eval_delta == 0:
            ep.append(epoch)
            loss_train.append(solver.compute_loss(Xtrain_eval, ytrain_eval) + 0.5 * lmbda * solver.compute_squared_norm())
            loss_test.append(solver.compute_loss(Xtest, ytest))
            acc_train.append((((2*ytrain - 1) * Xtrain.dot(solver.w)) >= 0).mean())
            acc_test.append((((2*ytest - 1) * Xtest.dot(solver.w)) >= 0).mean())

        if epoch == start_decay and (dropout_rate > 0 or isinstance(solver, solvers.SGD)):
            solver.start_decay()

        if dropout_rate > 0:
            if isinstance(solver, solvers.SVRG):  # heuristic from reviewer
                Xpert = Xtrain * random.binomial(1, 1 - dropout_rate, size=Xtrain.shape).astype(np.float32) / (1 - dropout_rate)
                solver.compute_snapshot(Xpert, ytrain)
                idxs = random.choice(n, n, p=q)
                t = time.time()
                solver.iterate_indexed(Xpert, ytrain, idxs)
            else:
                idxs = random.choice(n, n, p=q)
                Xtt = Xtrain[idxs]
                Xtt *= random.binomial(1, 1 - dropout_rate, size=Xtt.shape) / (1 - dropout_rate)
                t = time.time()
                solver.iterate(Xtt, ytrain[idxs], idxs)
        else:
            if isinstance(solver, solvers.SVRG): #  and epoch % 3 == 0:
                solver.compute_snapshot(Xtrain, ytrain)
            idxs = random.choice(n, n, p=q)
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
    solver = solvers.SGD(d, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss)
    return training(lmbda, dropout_rate, solver)

def train_sgd_avg(lmbda, dropout_rate, lr):
    solver = solvers.SGD(d, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss, average=True)
    return training(lmbda, dropout_rate, solver)

def train_sgd_nonu(lmbda, dropout_rate, lr):
    solver = solvers.SGD(d, lr=lr * (1 - dropout_rate)**2 / Lavg, lmbda=lmbda, loss=loss)
    q = np.asarray((Xtrain ** 2).sum(1)).flatten()
    q += q.mean()
    q /= q.sum()
    solver.set_q(q)
    return training(lmbda, dropout_rate, solver, q=q)

def train_miso(lmbda, dropout_rate, lr):
    solver = solvers.MISO(d, n, lmbda=lmbda, loss=loss)
    solver.decay(lr * min(1, n * lmbda * (1 - dropout_rate)**2 / Lmax))
    return training(lmbda, dropout_rate, solver)

def train_miso_avg(lmbda, dropout_rate, lr):
    solver = solvers.MISO(d, n, lmbda=lmbda, loss=loss, average=True)
    solver.decay(lr * min(1, n * lmbda * (1 - dropout_rate)**2 / Lmax))
    return training(lmbda, dropout_rate, solver)

def train_miso_nonu(lmbda, dropout_rate, lr):
    solver = solvers.MISO(d, n, lmbda=lmbda, loss=loss)
    q = np.asarray((Xtrain**2).sum(1)).flatten()
    q += q.mean()
    q /= q.sum()
    solver.set_q(q)
    solver.decay(lr * min(1, n * lmbda * (1 - dropout_rate)**2 / Lavg))
    return training(lmbda, dropout_rate, solver, q=q)

def train_saga(lmbda, dropout_rate, lr):
    solver = solvers.SAGA(d, n, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss)
    return training(lmbda, dropout_rate, solver)

def train_svrg(lmbda, dropout_rate, lr):
    solver = solvers.SVRG(d, n, lr=lr * (1 - dropout_rate)**2 / Lmax, lmbda=lmbda, loss=loss)
    return training(lmbda, dropout_rate, solver)

train_fn = {'sgd': train_sgd,
            'sgd_avg': train_sgd_avg,
            'sgd_nonu': train_sgd_nonu,
            'miso': train_miso,
            'miso_avg': train_miso_avg,
            'miso_nonu': train_miso_nonu,
            'saga': train_saga,
            'svrg': train_svrg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dropout training on breast cancer dataset')
    parser.add_argument('--num-workers', default=1,
                        type=int, help='number of threads for grid search')
    parser.add_argument('--pdf-file', default=None, help='pdf file to save to')
    parser.add_argument('--pkl-file', default=None, help='pickle file to save to')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    args = parser.parse_args()
    seed = args.seed
    print('seed:', seed)

    logging.info('loading gene data')
    Xtrain, ytrain, Xtest, ytest = load_breast_cancer()
    n = Xtrain.shape[0]
    d = Xtrain.shape[1]

    Lmax = (Xtrain**2).sum(1).max()
    Lavg = (Xtrain**2).sum(1).mean()

    pp = None
    if args.pdf_file:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.backends.backend_pdf import PdfPages
        import curves
        pp = PdfPages(args.pdf_file)

    pkl = None
    if args.pkl_file is not None:
        pkl = []

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
            if pp or pkl is not None:
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

                if pkl is not None:
                    pkl.append(((lmbda, delta), plot_res))
                if pp:
                    curves.plot_loss(plot_res, ty='train', log=True, step=1, last=None,
                                     small=False, legend=True, title='gene dropout, $\delta$ = {:.2f}'.format(delta))
                    pp.savefig()

    if pp:
        pp.close()
    if pkl:
        import pickle
        pickle.dump(pkl, open(args.pkl_file, 'wb'))

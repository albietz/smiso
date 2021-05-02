import numpy as np
from sklearn.feature_extraction import image


def whiten_images(X, verbose=True, patch_size=3):
    '''X: images, shape (num_images, h, w, num_channels).'''
    h, w, c = X.shape[1:]
    for idx in range(X.shape[0]):
        if verbose and idx % 1000 == 0:
            print(idx)
        im = X[idx]
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        if p.ndim < 4:
            p = p[:,:,:,None]
        p -= p.mean((1,2))[:,None,None,:]
        im = image.reconstruct_from_patches_2d(p, (h, w, c))
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        p = p.reshape(p.shape[0], -1)

        cov = p.T.dot(p)
        s, U = np.linalg.eigh(cov)
        s[s < 0] = 0
        s = np.sqrt(s)
        ind = s < 1e-8 * s.max()
        s = 1. / np.sqrt(s)  # reduce whitening
        s[ind] = 0

        p = p.dot(U.dot(np.diag(s)).dot(U.T))
        p = p.reshape(p.shape[0], patch_size, patch_size, -1)
        X[idx] = image.reconstruct_from_patches_2d(p, (h, w, c))

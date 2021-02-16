import numpy as np
import sklearn
import sklearn.decomposition

def pca(arr, method='pca', ncomp=None):
    """
    Runs principle component analysis on the input array.

    Arguments
    ---------
    arr: 2D array
        (m, n) array, where n is the size of the dataset (e.g., times
        in an observation) and m is the number of vectors

    Returns
    -------
    evalues: 1D array
        array of eigenvalues of size n

    evectors: 2D array
        (m, m) array of sorted eigenvectors

    proj: 2D array
        (m, n) array of data projected in the new space

    Notes
    -----
    See https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
    """
    nt = arr.shape[1]
    
    if method == 'pca':
        arr = arr.T
        # Subtract the mean
        m = (arr - np.mean(arr.T, axis=1)).T
        #m = arr
        # Compute eigenvalues
        evalues, evectors = np.linalg.eig(np.cov(m))
        # Sort descending
        idx = np.argsort(evalues)[::-1]
        evalues  = evalues[   idx]
        evectors = evectors[:,idx]
        # Calculate projection of the data in the new space
        proj = np.dot(evectors.T, m)

    elif method == 'tsvd':
        tpca = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
        tpca.fit(arr.T)
        evalues  = tpca.explained_variance_
        evectors = tpca.components_
        proj = np.zeros((ncomp, nt))
        for i in range(ncomp):
            proj[i] = np.sum(evectors[i] * arr.T, axis=1)

        evectors = evectors.T
            
        
    return evalues, evectors, proj

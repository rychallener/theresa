import numpy as np

def pca(arr):
    """
    Runs principle component analysis on the input array.

    Arguments
    ---------
    arr: 2D array
        (m, n) array, where m is the size of the dataset (e.g., times
        in an observation) and n is the number of vectors

    Returns
    -------
    evalues: 1D array
        array of eigenvalues of size n

    evectors: 2D array
        (n, n) array of sorted eigenvectors

    proj: 2D array
        (m, n) array of data projected in the new space

    Notes
    -----
    See https://glowingpython.blogspot.com/2011/07/pca-and-image-compression-with-numpy.html
    """
    arr = arr.T
    # Subtract the mean
    m = (arr - np.mean(arr.T, axis=1)).T
    # Compute eigenvalues
    evalues, evectors = np.linalg.eig(np.cov(m))
    # Sort descending
    idx = np.argsort(evalues)[::-1]
    evalues  = evalues[   idx]
    evectors = evectors[:,idx]
    # Calculate projection of the data in the new space
    proj = np.dot(evectors.T, m)
    return evalues, evectors, proj

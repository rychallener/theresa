import numpy as np

def pca(arr):
    """
    """
    
    # Subtract the mean
    m = (arr - np.mean(arr.T, axis=1)).T
    # Compute eigenvalues
    evalues, evectors = np.linalg.eig(np.cov(m))
    # Sort descending
    idx = np.argsort(evalues)[::-1]
    evalues  = evalues[   idx]
    evectors = evectors[:,idx]
    # Calculate projection of the data in the new space
    score = np.dot(evectors, m)
    return evectors, score, evalues

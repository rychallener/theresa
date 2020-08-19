import numpy as np
import pickle

def load_fit(fname):
    """
    Load a fit object from a pickled file.
    """
    with open(fname, 'rb') as f:
        fit = pickle.load(f)

    return fit

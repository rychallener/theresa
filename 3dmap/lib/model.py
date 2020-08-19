import os
import numpy as np
import pickle

class Fit:
    """
    A class to hold attributes and methods related to fitting a model
    or set of models to data.
    """
    def save(self, outdir, fname=None):
        # Note: starry objects are not pickleable, so they
        # cannot be added to the Fit object as attributes. Possible
        # workaround by creating a custom Pickler?
        if type(fname) == type(None):
            fname = 'fit.pkl'

        with open(os.path.join(outdir, fname), 'wb') as f:
            pickle.dump(self, f)

def fit_2d(params, ecurves, t, y00, sflux, ncurves):
    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * params[i]
   
    f += params[i+1] * y00

    f += params[i+2]

    f += sflux

    return f

def fit_3d():
    pass
    

import numpy as np

def fit_2d(ecurves, y00, t, params, ncurves):
    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * getattr(params, 'e{}'.format(i))
   
    f += y00

    f += params.scorr

    return f

def fit_3d():
    pass
    

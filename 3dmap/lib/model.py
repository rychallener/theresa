import numpy as np

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
    

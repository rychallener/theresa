import numpy as np

def fit_2d(params, ecurves, t, y00, sflux, ncurves):
    """
    Basic 2D fitting routine for a single wavelength.
    """
    f = np.zeros(len(t))

    for i in range(ncurves):
        f += ecurves[i] * params[i]
   
    f += params[i+1] * y00

    f += params[i+2]

    f += sflux

    return f

def fit_2d_wl(params, ecurves, t, wl, y00, sflux, ncurves):
    """
    2D fitting driver that calls the 2D fitting routine for each
    wavelength.
    """
    f = np.zeros(len(t) * len(wl))

    nt   = len(t)
    nw   = len(wl)
    npar = int(len(params) / nw) # params per wavelength
    for i in range(nw):
        f[i*nt:(i+1)*nt] = fit_2d(params[i*npar:(i+1)*npar], ecurves,
                                  t, y00, sflux, ncurves)

    return f
                                        
    
    

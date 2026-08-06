import numpy as np
import utils
import scipy.interpolate as sci
import time
from numba import njit
from numba import  jit

def contribution(tgrid, wn, taugrid, p):
    nlev, ncolumn = tgrid.shape
    nwn = len(wn)
    
    cf = np.zeros((ncolumn, nlev, nwn))

    # Pressure is always the same. Calculate out of the loop
    # Skip bottom layer (99 gradients and 100 cells -- have to put
    # skip somewhere) since bottom layer should be invisible
    # anyway
    dlp = np.zeros((nlev, 1))
    for k in range(nlev-1, 0, -1):
        dlp[k] = np.log(p[k-1]) - np.log(p[k])

    for i in range(ncolumn):
        bb = utils.blackbody(tgrid[:,i], wn)
        trans = np.exp(-taugrid[i])
        dt = np.zeros((nlev, nwn))

        # Skip bottom layer (leave as 0s)
        for k in range(nlev-1, 0, -1):
            dt[k] = trans[k] - trans[k-1]

        cf[i] = bb * dt / dlp
        # Replace division-by-zero NaNs with zero
        cf[i,0,:] = 0.0

    return cf

def contribution_filters(tgrid, wn, taugrid, p, filtwn, filttrans):
    nlev, ncolumn = np.shape(tgrid)
    nwn = len(wn)
    nfilt = len(filtwn)

    cf = contribution(tgrid, wn, taugrid, p)

    # Filter-integrated contribution functions
    filter_cf = np.zeros((ncolumn, nlev, nfilt))

    for i in range(nfilt):
        # Interpolate filter to spectrum resolution. Assume zero
        # transmission out-of-bounds.
        interp = sci.interp1d(filtwn[i], filttrans[i], bounds_error=False,
                              fill_value=0.0)
        interptrans = interp(wn)
        integtrans  = np.trapz(interptrans)
        
        # Contribution functions convolved with filter transmissions
        cf_trans = cf * interptrans

        # Integrate
        for j in range(ncolumn):
            filter_cf[j,:,i] = \
                np.trapz(cf_trans[j], axis=1) / integtrans
        
    return filter_cf

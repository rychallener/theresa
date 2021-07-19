import numpy as np
import utils
import scipy.interpolate as sci
import time
from numba import njit
from numba import  jit

def contribution(tgrid, wn, taugrid, p):
    nlev, nlat, nlon = tgrid.shape
    nwn = len(wn)
    
    cf = np.zeros((nlat, nlon, nlev, nwn))

    # Pressure is always the same. Calculate out of the loop
    # Start with second from top as delta-log-p makes no sense
    # for the top layer (just leave at 0)
    dlp = np.zeros((nlev, 1))
    for k in range(nlev-2, -1, -1):
        dlp[k] = np.log(p[k]) - np.log(p[k+1])

    for i in range(nlat):
        for j in range(nlon):
            bb = utils.blackbody(tgrid[:,i,j], wn)
            trans = np.exp(-taugrid[i,j])
            dt = np.zeros((nlev, nwn))
            
            # Skip top layer (leave as 0s)
            for k in range(nlev-2, -1, -1):
                dt[k] = trans[k+1] - trans[k]

            cf[i,j] = bb * dt / dlp
            # Replace division-by-zero NaNs with zero
            cf[i,j,nlev-1,:] = 0.0

    return cf

def contribution_filters(tgrid, wn, taugrid, p, filtwn, filttrans):
    nlev, nlat, nlon = np.shape(tgrid)
    nwn = len(wn)
    nfilt = len(filtwn)

    cf = contribution(tgrid, wn, taugrid, p)

    # Filter-integrated contribution functions
    filter_cf = np.zeros((nlat, nlon, nlev, nfilt))

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
        for j in range(nlat):
            for k in range(nlon):
                filter_cf[j,k,:,i] = \
                    np.trapz(cf_trans[j,k], axis=1) / integtrans
        
    return filter_cf

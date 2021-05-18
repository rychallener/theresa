import numpy as np
import utils
import scipy.interpolate as sci

def contribution(tgrid, wn, taugrid, p):
    nlev, nlat, nlon = tgrid.shape
    nwn = len(wn)
    
    cf = np.zeros((nlat, nlon, nlev, nwn))

    for i in range(nlat):
        for j in range(nlon):
            bb = utils.blackbody(tgrid[:,i,j], wn)
            trans = np.exp(-taugrid[i,j])
            dlp = np.zeros(nlev)
            dt = np.zeros((nlev, nwn))
            for k in range(nlev-1, -1, -1):
                if k == nlev - 1:
                    dt[k,:]     = 0.0
                    dlp[k]      = 0.0
                    cf[i,j,k,:] = 0.0
                else:
                    dt[k,:] = trans[k+1] - trans[k]
                    dlp[k]  = np.log(p[k]) - np.log(p[k+1])
                    cf[i,j,k] = bb[k] * dt[k,:] / dlp[k]

    return cf

def contribution_filters(tgrid, wn, taugrid, p, filtwn, filttrans):
    nlev, nlat, nlon = tgrid.shape
    nwn = len(wn)
    nfilt = len(filtwn)

    cf = contribution(tgrid, wn, taugrid, p)

    # Filter-integrated contribution functions
    filter_cf = np.zeros((nlat, nlon, nlev, nfilt))
    # Contribution functions convolved with filter transmissions
    cf_trans = np.zeros((nlat, nlon, nlev, nwn))

    for i in range(nfilt):
        # Interpolate filter to spectrum resolution. Assume zero
        # transmission out-of-bounds.
        interp = sci.interp1d(filtwn[i], filttrans[i], bounds_error=False,
                              fill_value=0.0)
        interptrans = interp(wn)
        
        # Integrate
        for j in range(nlat):
            for k in range(nlon):
                cf_trans[j,k,:,:] = cf[j,k,:,:] * interptrans
                for l in range(nlev):
                    filter_cf[j,k,l,i] = \
                        np.trapz(cf_trans[j,k,l]) / np.trapz(interptrans)

    return filter_cf

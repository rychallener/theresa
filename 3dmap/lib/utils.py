import numpy as np
import pickle
import theano
import time
import constants as c
import scipy.interpolate as spi
import starry

def initsystem(fit):
    cfg = fit.cfg

    star = starry.Primary(starry.Map(ydeg=1, amp=1),
                          m   =cfg.star.m,
                          r   =cfg.star.r,
                          prot=cfg.star.prot)

    planet = starry.kepler.Secondary(starry.Map(ydeg=cfg.lmax),
                                     m    =cfg.planet.m,
                                     r    =cfg.planet.r,
                                     porb =cfg.planet.porb,
                                     prot =cfg.planet.prot,
                                     Omega=cfg.planet.Omega,
                                     ecc  =cfg.planet.ecc,
                                     w    =cfg.planet.w,
                                     t0   =cfg.planet.t0,
                                     inc  =cfg.planet.inc,
                                     theta0=180)

    system = starry.System(star, planet)

    return star, planet, system

def specint(wn, spec, filtwn_list, filttrans_list):
    """
    Integrate a spectrum over the given filters.

    Arguments
    ---------
    wn: 1D array
        Wavenumbers (/cm) of the spectrum

    spec: 1D array
        Spectrum to be integrated

    filtwn_list: list
        List of arrays of filter wavenumbers, in /cm.

    filttrans_list: list
        List of arrays of filter transmission. Same length as filtwn_list.

    Returns
    -------
    intspec: 1D array
        The spectrum integrated over each filter. 
    """
    if len(filtwn_list) != len(filttrans_list):
        print("ERROR: list sizes do not match.")
        raise Exception
    
    intspec = np.zeros(len(filtwn_list)) 
    
    for i, (filtwn, filttrans) in enumerate(zip(filtwn_list, filttrans_list)):
        # Sort ascending
        idx = np.argsort(filtwn)
        
        intfunc = spi.interp1d(filtwn[idx], filttrans[idx],
                               bounds_error=False, fill_value=0)

        # Interpolate transmission
        inttrans = intfunc(wn)

        # Normalize to one
        norminttrans = inttrans / np.trapz(inttrans, wn)

        # Integrate filtered spectrum
        intspec[i] = np.trapz(spec * norminttrans, wn)

    return intspec

    
def vislon(planet, fit):
    """
    Determines the range of visible longitudes based on times of
    observation.

    Arguments
    ---------
    planet: starry Planet object
        Planet object

    fit: Fit object
        Fit object. Must contain observation information.

    Returns
    -------
    minlon: float
        Minimum visible longitude, in degrees

    maxlon: float
        Maximum visible longitude, in degrees
    """
    t = fit.t

    porb   = planet.porb   # days / orbit
    prot   = planet.prot   # days / rotation
    t0     = planet.t0     # days
    theta0 = planet.theta0 # degrees

    # Central longitude at each time ("sub-observer" point)
    centlon = theta0 - (t - t0) / prot * 360

    # Minimum and maximum longitudes (assuming +/- 90 degree
    # visibility)
    limb1 = centlon - 90
    limb2 = centlon + 90

    # Rescale to [-180, 180]
    limb1 = (limb1 + 180) % 360 - 180
    limb2 = (limb2 + 180) % 360 - 180

    return np.min(limb1.eval()), np.max(limb2.eval())
    
    
def readfilters(filterfiles):
    """
    Reads filter files and determines the mean wavelength.
    
    Arguments
    ---------
    filterfiles: list
        list of paths to filter files

    Returns
    -------
    filtmid: 1D array
        Array of mean wavelengths
    """
    filtwl_list    = []
    filtwn_list    = []
    filttrans_list = []
    
    wnmid = np.zeros(len(filterfiles))
    for i, filterfile in enumerate(filterfiles):
        filtwl, trans = np.loadtxt(filterfile, unpack=True)
        
        filtwn = 1.0 / (filtwl * c.um2cm)

        wnmid[i] = np.sum(filtwn * trans) / np.sum(trans)

        filtwl_list.append(filtwl)
        filtwn_list.append(filtwn)
        filttrans_list.append(trans)

    wlmid = 1 / (c.um2cm * wnmid)

    return filtwl_list, filtwn_list, filttrans_list, wnmid, wlmid
        
def visibility(t, latgrid, longrid, dlat, dlon, theta0, prot, t0, rp,
               rs, x, y):
    """
    Calculate the visibility of a grid of cells on a planet
    at a specific time. Returns a combined visibility based on the
    observer's line-of-sight and the effect of the star.
    """
    if latgrid.shape != longrid.shape:
        print("Number of latitudes and longitudes do not match.")
        raise Exception

    losvis  = np.zeros(latgrid.shape)
    starvis = np.zeros(latgrid.shape)
    
    # Flag to do star visibility calculation (improves efficiency)
    dostar = True

    # Central longitude (observer line-of-sight)
    centlon = theta0 - (t - t0) / prot * 2 * np.pi

    # Convert relative to substellar point
    centlon = (centlon + np.pi) % (2 * np.pi) - np.pi
    
    xsep = x[0] - x[1]
    ysep = y[0] - y[1]
    d = np.sqrt(xsep**2 + ysep**2)

    # Visible fraction due to star        
    # No grid cells visible. Return 0s
    if (d < rs - rp):
        return np.zeros(latgrid.shape)
    
    # All grid cells visible. No need to do star calculation.
    elif (d > rs + rp):
        starvis[:,:] = 1.0
        dostar     = False
    # Otherwise, time is during ingress/egress and we cannot simplify
    # calculation

    nlat, nlon = latgrid.shape
    for i in range(nlat):
        for j in range(nlon):
            # Angles wrt the observer
            lat = latgrid[i,j]
            lon = longrid[i,j]
            
            phi   = lon - centlon
            theta = lat
            phimin   = phi - dlon / 2.
            phimax   = phi + dlon / 2.

            thetamin = lat - dlat / 2.
            thetamax = lat + dlat / 2.

            # Cell is not visible at this time. No need to calculate further.
            if (phimin > np.pi / 2.) or (phimax < -np.pi / 2.):
                losvis[i,j] = 0

            # Cell is visible at this time
            else:
                # Determine visible phi/theta range of the cell
                phirng   = (np.max((phimin,   -np.pi / 2.)),
                            np.min((phimax,    np.pi / 2.)))
                thetarng = (np.max((thetamin, -np.pi / 2.)),
                            np.min((thetamax,  np.pi / 2.)))
                # Mean visible latitude/longitude
                thetamean = np.mean(thetarng)
                phimean   = np.mean(phirng)

                # Visibility based on LoS
                losvis[i,j] = np.cos(thetamean) * np.cos(phimean)

                # Grid cell maybe only partially visible
                if dostar:
                    # Grid is "within" the star
                    if dgrid(x, y, rp, thetamean, phimean) < rs:
                        starvis[i,j] = 0.0
                    # Grid is not in the star
                    else:
                        starvis[i,j] = 1.0

    return starvis * losvis

def dgrid(x, y, rp, theta, phi):
    """
    Calculates the projected distance between a latitude (theta) and a 
    longitude (phi) on a planet with radius rp to a star. Projected
    star position is (x[0], y[0]) and planet position is (x[1], y[1]).
    """
    xgrid = x[1] + rp * np.cos(theta) * np.sin(phi)
    ygrid = y[1] + rp * np.sin(theta)
    d = np.sqrt((xgrid - x[0])**2 + (ygrid - y[0])**2)
    return d

def t_dgrid():
    """
    Returns a theano function of dgrid(), with the same arguments.
    """
    print('Defining theano function.')
    arg1 = theano.tensor.dvector('x')
    arg2 = theano.tensor.dvector('y')
    arg3 = theano.tensor.dscalar('rp')
    arg4 = theano.tensor.dscalar('theta')
    arg5 = theano.tensor.dscalar('phi')

    f = theano.function([arg1, arg2, arg3, arg4, arg5],
                        dgrid(arg1, arg2, arg3, arg4, arg5))    
    return f

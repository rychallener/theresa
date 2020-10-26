import numpy as np
import pickle
import theano
import scipy.interpolate as spi

def specint(wn, spec, filtfiles):
    """
    Integrate a spectrum over the given filters.

    Arguments
    ---------
    wn: 1D array
        Wavenumbers (/cm) of the spectrum

    spec: 1D array
        Spectrum to be integrated

    filtfiles: list
        Paths to filter files, which should be 2 columns.
        First column is wavelength (microns), and second column
        is filter throughput.

    Returns
    -------
    intspec: 1D array
        The spectrum integrated over each filter. 
    """
    intspec = np.zeros(len(filtfiles)) 
    
    for i, filtfile in enumerate(filtfiles):
        filtwl, trans = np.loadtxt(filtfile, unpack=True)

        um2cm = 1e-4
        
        filtwn = 1.0 / (filtwl * um2cm)

        # Sort ascending
        idx = np.argsort(filtwn)
        filtwn = filtwn[idx]
        trans  = trans[idx]
        
        intfunc = spi.interp1d(filtwn, trans, bounds_error=False,
                               fill_value=0)

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
    
    
def filtmean(filterfiles):
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
    wnmid = np.zeros(len(filterfiles))
    for i, filterfile in enumerate(filterfiles):
        filtwl, trans = np.loadtxt(filterfile, unpack=True)
        
        um2cm = 1e-4
        
        filtwn = 1.0 / (filtwl * um2cm)

        wnmid[i] = np.sum(filtwn * trans) / np.sum(trans)

    wlmid = 1 / (um2cm * wnmid)

    return wlmid
        
def visibility(t, latgrid, longrid, dlat, dlon, planet, system):
    """
    Calculate the visibility of a grid of cells on a planet
    at a specific time. Returns a combined visibility based on the
    observer's line-of-sight and the effect of the star.
    """
    if len(latgrid) != len(longrid):
        print("Numbers of latitudes and longitudes do not match.")
        raise Exception

    losvis  = np.zeros(len(latgrid))
    starvis = np.zeros(len(latgrid))
    
    # Flag to do star visibility calculation (improves efficiency)
    dostar = True
    
    theta0 = planet.theta0.eval()
    prot   = planet.prot.eval()
    t0     = planet.t0.eval()

    rs = system.bodies[0].r
    rp = system.bodies[1].r

    # Central longitude (observer line-of-sight)
    centlon = theta0 - (t - t0) / prot * 360

    # Convert relative to substellar point
    centlon = (centlon + 180) % 360 - 180

    # Convert to radians
    centlon *= np.pi / 180.

    dlat *= np.pi / 180.
    dlon *= np.pi / 180.
    
    x, y, z = system.position(t)
    xsep = x[0] - x[1]
    ysep = y[0] - y[1]
    d = np.sqrt(xsep**2 + ysep**2)

    # Visible fraction due to star        
    # No grid cells visible. Return 0s
    if (d < rs - rp).eval():
        return np.zeros(len(latgrid))
    
    # All grid cells visible. No need to do star calculation.
    elif (d > rs + rp).eval():
        starvis[:] = 1.0
        dostar = False
    # Otherwise, time is during ingress/egress and we cannot simplify
    # calculation

    for igrid, (lat, lon) in enumerate(zip(latgrid, longrid)):
        # Angles wrt the observer
        phi   = lon - centlon
        theta = lat
        phimin   = phi - dlon / 2.
        phimax   = phi + dlon / 2.

        thetamin = lat - dlat / 2.
        thetamax = lat + dlat / 2.

        # Cell is not visible at this time. No need to calculate further.
        if (phimin > np.pi / 2.) or (phimax < -np.pi / 2.):
            losvis[igrid] = 0
            
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
            losvis[igrid] = np.cos(thetamean) * np.cos(phimean)

            # Grid cell maybe only partially visible
            if dostar:
                # Grid is "within" the star
                if instar(x, y, rp, rs, thetamean, phimean).eval():
                    starvis[igrid] = 0.0
                # Grid is not in the star
                else:
                    starvis[igrid] = 1.0

    return starvis * losvis

def instar(x, y, rp, rs, theta, phi):
    xgrid = x[1] + rp * np.cos(theta) * np.sin(phi)
    ygrid = y[1] + rp * np.sin(theta)
    dgrid = np.sqrt((xgrid - x[0])**2 + (ygrid - y[0])**2)
    return dgrid < rs
                    

import numpy as np
import pickle
import theano
import time
import constants as c
import scipy.constants as sc
import scipy.interpolate as spi
import eigen
import starry
import progressbar
import theano
import theano.tensor as tt
import mc3.stats as ms
from numba import njit

def initsystem(fit, ydeg):
    '''
    Uses a fit object to build the respective starry objects. Useful
    because starry objects cannot be pickled. Returns a tuple of
    (star, planet, system).
    '''
    
    cfg = fit.cfg

    star = starry.Primary(starry.Map(ydeg=1, amp=1),
                          m   =cfg.star.m,
                          r   =cfg.star.r,
                          prot=cfg.star.prot)

    planet = starry.kepler.Secondary(starry.Map(ydeg=ydeg),
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
        
def visibility(t, latgrid, longrid, dlatgrid, dlongrid, theta0, prot,
               t0, rp, rs, x, y):
    """
    Calculate the visibility of a grid of cells on a planet at a specific
    time. Returns a combined visibility based on the observer's
    line-of-sight, the area of the cells, and the effect of the star.

    Arguments
    ---------
    t: float
        Time to calculate visibility.
    
    latgrid: 2D array
        Array of latitudes, in radians, from -pi/2 to pi/2.

    longrid: 2D array
        Array of longitudes, in radians, from -pi to pi.

    dlat: float
        Latitude resolution in radians.

    dlon: float
        Longitude resoltuion in radians.

    theta0: float
        Rotation at t0 in radians.

    prot: float
        Rotation period, the same units as t.

    t0: float
        Time of transit, same units as t.

    rp: float
        Planet radius in solar radii.

    rs: float
        Star radius in solar radii.

    x: tuple
        x position of (star, planet)

    y: tuple
        y position of (star, planet)

    Returns
    -------
    vis: 2D array
        Visibility of each grid cell. Same shape as latgrid and longrid.

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
            lat  = latgrid[i,j]
            lon  = longrid[i,j]
            dlat = dlatgrid[i,j]
            dlon = dlongrid[i,j]
            
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
                phirng   = np.array((np.max((phimin,   -np.pi / 2.)),
                                     np.min((phimax,    np.pi / 2.))))
                thetarng = np.array((np.max((thetamin, -np.pi / 2.)),
                                     np.min((thetamax,  np.pi / 2.))))


                # Visibility based on LoS
                # This is the integral of
                #
                # A(theta, phi) V(theta, phi) dtheta dphi
                #
                # where
                #
                # A = r**2 cos(theta)
                # V = cos(theta) cos(phi)
                #
                # Here we've normalized by pi*r**2, since
                # visibility will be applied to Fp/Fs where planet
                # size is already taken into account.
                losvis[i,j] = (np.diff(thetarng/2) + \
                               np.diff(np.sin(2*thetarng) / 4)) * \
                    np.diff(np.sin(phirng)) / \
                    np.pi

                # Grid cell maybe only partially visible
                if dostar:
                    thetamean = np.mean(thetarng)
                    phimean   = np.mean(phirng)
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

def mapintensity(map, lat, lon, amp):
    """
    Calculates a grid of intensities, multiplied by the amplitude given.
    """
    grid = map.intensity(lat=lat.flatten(), lon=lon.flatten()).eval()
    grid *= amp
    grid = grid.reshape(lat.shape)
    return grid


def hotspotloc_driver(fit, map):
    """
    Calculates a distribution of hotspot locations based on the MCMC
    posterior distribution.

    Note that this function assumes the first ncurves parameters
    in the posterior are associated with eigencurves. This will not
    be true if some eigencurves are skipped over, as MC3 does not
    include fixed parameters in the posterior.

    Inputs
    ------
    fit: Fit instance

    map: Map instance (not starry Map)

    Returns
    -------
    hslocbest: tuple
        Best-fit hotspot location (lat, lon), in degrees.

    hslocstd: tuple
        Standard deviation of the hotspot location posterior distribution
        as (lat, lon)

    hspot: tuple
        Marginalized posterior distributions of latitude and longitude
    """
    
    post = map.post[map.zmask]

    nsamp, nfree = post.shape

    ntries     =  5
    oversample =  1

    if fit.cfg.twod.ncalc > nsamp:
        print("Warning: ncalc reduced to match burned-in sample.")
        ncalc = nsamp
    else:
        ncalc = fit.cfg.twod.ncalc
    
    hslon = np.zeros(ncalc)
    hslat = np.zeros(ncalc)
    thinning = nsamp // ncalc

    bounds = (-45, 45),(fit.minvislon, fit.maxvislon)
    smap = starry.Map(ydeg=map.lmax)
    # Function defined in this way to avoid passing non-numeric arguments
    def hotspotloc(yval):       
        smap[1:,:] = yval
        lat, lon, val = smap.minimize(oversample=oversample,
                                      ntries=ntries, bounds=bounds)
        return lat, lon, val

    arg1 = tt.dvector()
    t_hotspotloc = theano.function([arg1], hotspotloc(arg1))

    # Note the maps created here do not include the correct uniform
    # component because that does not affect the location of the
    # hotspot. Also note that the eigenvalues are negated because
    # we want to maximize, not minize, but starry only includes
    # a minimize method.
    pbar = progressbar.ProgressBar(max_value=ncalc)
    for i in range(0, ncalc):
        ipost = i * thinning
        yval = np.zeros((map.lmax+1)**2-1)
        for j in range(map.ncurves):
            yval += -1 * post[ipost,j] * map.eigeny[j,1:]

        hslat[i], hslon[i], _ = t_hotspotloc(yval)
        pbar.update(i+1)

    star, planet, system = initsystem(fit, map.lmax)
    planet.map[1:,:] = 0.0
    for j in range(map.ncurves):
        planet.map[1:,:] += -1 * map.bestp[j] * map.eigeny[j,1:]
    hslatbest, hslonbest, _ = planet.map.minimize(oversample=oversample,
                                                  bounds=bounds,
                                                  ntries=ntries)
    hslonbest = hslonbest.eval()
    hslatbest = hslatbest.eval()

    hslonstd = np.std(hslon)
    hslatstd = np.std(hslat)

    # Two-sided errors 
    pdf, xpdf, hpdmin = ms.cred_region(hslon)
    crlo = np.amin(xpdf[pdf>hpdmin])
    crhi = np.amax(xpdf[pdf>hpdmin])
    hsloncrlo = crlo - hslonbest
    hsloncrhi = crhi - hslonbest

    pdf, xpdf, hpdmin = ms.cred_region(hslat)
    crlo = np.amin(xpdf[pdf>hpdmin])
    crhi = np.amax(xpdf[pdf>hpdmin])
    hslatcrlo = crlo - hslatbest
    hslatcrhi = crhi - hslatbest

    hslocbest  = (hslatbest, hslonbest)
    hslocstd   = (hslatstd,  hslonstd)
    hslocpost  = (hslat,     hslon)
    hsloctserr = ((hslatcrhi, hslatcrlo), (hsloncrhi, hsloncrlo))
    
    return hslocbest, hslocstd, hslocpost, hsloctserr

def tmappost(fit, map):
    post = map.post[map.zmask]

    nsamp, nfree = post.shape
    ncurves = map.ncurves

    if fit.cfg.twod.ncalc > nsamp:
        print("Warning: ncalc reduced to match burned-in sample.")
        ncalc = nsamp
    else:
        ncalc = fit.cfg.twod.ncalc

    thinning = nsamp // ncalc

    fmaps = np.zeros((ncalc, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    tmaps = np.zeros((ncalc, fit.cfg.twod.nlat, fit.cfg.twod.nlon))
    
    star, planet, system = initsystem(fit, map.lmax)

    def calcfmap(yval, unifamp):
        planet.map[1:,:] = 0.0
        amp = unifamp - 1
        fmap = planet.map.intensity(lat=fit.lat.flatten(),
                                    lon=fit.lon.flatten()) * amp

        planet.map[1:,:] = yval
        fmap += planet.map.intensity(lat=fit.lat.flatten(),
                                     lon=fit.lon.flatten())

        return fmap

    arg1 = tt.dvector()
    arg2 = tt.dscalar()
    t_calcfmap = theano.function([arg1, arg2], calcfmap(arg1, arg2))
        
    pbar = progressbar.ProgressBar(max_value=ncalc)
    for i in range(ncalc):
        ipost = i * thinning
        yval = np.zeros((map.lmax+1)**2-1)
        for j in range(map.ncurves):
            yval += post[ipost,j] * map.eigeny[j,1:]
            
        fmaps[i] = t_calcfmap(yval, post[ipost, ncurves]).reshape(fit.lat.shape)
        tmaps[i] = fmap_to_tmap(fmaps[i], map.wlmid*1e-6,
                                fit.cfg.planet.r, fit.cfg.star.r,
                                fit.cfg.star.t, post[ipost,ncurves+1])
        
        pbar.update(i+1)

    return fmaps, tmaps

def fmap_to_tmap(fmap, wl, rp, rs, ts, scorr):
    '''
    Convert flux map to brightness temperatures.
    See Rauscher et al., 2018, eq. 8
    '''
    ptemp = (sc.h * sc.c) / (wl * sc.k)
    sfact = 1 + scorr
    tmap = ptemp / np.log(1 + (rp / rs)**2 *
                          (np.exp(ptemp / ts) - 1) /
                          (np.pi * fmap * sfact))
    return tmap

def ess(chain):
    '''
    Calculates the Steps Per Effectively-Independent Sample and
    Effective Sample Size (ESS) of a chain from an MCMC posterior 
    distribution.

    Adapted from some code I wrote for MC3 many years ago, and
    the SPEIS/ESS calculation in BART.
    '''
    nciter, npar = chain.shape

    speis = np.zeros(npar)
    ess   = np.zeros(npar)

    for i in range(npar):
        mean     = np.mean(chain[:,i])
        autocorr = np.correlate(chain[:,i] - mean,
                                chain[:,i] - mean,
                                mode='full')
        # Keep lags >= 0 and normalize
        autocorr = autocorr[np.size(autocorr) // 2:] / np.max(autocorr)
        # Sum adjacent pairs (Geyer, 1993)
        pairsum = autocorr[:-1:2] + autocorr[1::2]
        # Find where the sum goes negative, or use the whole thing
        if np.any(pairsum > 0):
            idx = np.where(pairsum < 0)[0][0]
        else:
            idx = len(pairsum)
            print("WARNING: parameter {} did not decorrelate!"
                  "Do not trust ESS/SPEIS!".format(i))
        # Calculate SPEIS
        speis[i] = -1 + 2 * np.sum(pairsum[:idx])
        ess[i]   = nciter / speis[i]

    return speis, ess

def crsig(ess, cr=0.683):
    '''
    Calculates the absolute error on an estimate of a credible region
    of a given percentile based on the effective sample size.

    See Harrington et al, 2021.

    Arguments
    ---------
    ess: int
        Effective Sample Size

    cr: float
        Credible region percentile to calculate error on. E.g., 
        for a 1-sigma region, use 0.683 (the default).

    Returns
    -------
    crsig: float
        The absolute error on the supplied credible region.
    '''
    return (cr * (1 - cr) / (ess + 3))**0.5

@njit
def fast_linear_interp(a, b, x):
    return (b[1] - a[1]) / (b[0] - a[0]) * (x - a[0]) + a[1]

@njit
def blackbody(T, wn):
    '''
    Calculates the Planck function for a grid of temperatures and
    wavenumbers. Wavenumbers must be in /cm.
    '''
    nt  = len(T)
    nwn = len(wn)
    bb = np.zeros((nt, nwn))

    # Convert from /cm to /m
    wn_m = wn * 1e2
    for i in range(nt):
        bb[i] = (2.0 * sc.h * sc.c**2 * wn_m**3) \
            * 1/(np.exp(sc.h * sc.c * wn_m / sc.k / T[i]) - 1.0)

    return bb    

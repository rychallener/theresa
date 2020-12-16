import numpy as np
import pca
import utils
import scipy.constants as sc

def mkcurves(system, t, lmax):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax.

    Arguments
    ---------
    system: object
        A starry system object, initialized with a star and a planet

    t: 1D array
        Array of times at which to calculate eigencurves

    lmax: integer
        Maximum l to use in spherical harmonic maps

    Returns
    -------
    eigeny: 2D array
        nharm x ny array of y coefficients for each harmonic. nharm is
        the number of harmonics, including positive and negative versions
        and excluding Y00. That is, 2 * ((lmax + 1)**2 - 1). ny is the
        number of y coefficients to describe a harmonic with degree lmax.
        That is, (lmax + 1)**2.

    evalues: 1D array
        nharm length array of eigenvalues

    evectors: 2D array
        nharm x nt array of normalized (unit) eigenvectors

    proj: 2D array
        nharm x nt array of the data projected in the new space (the PCA
        "eigencurves"). The imaginary part is discarded, if nonzero.
    """
    star   = system.bodies[0]
    planet = system.bodies[1]

    nt = len(t)
    
    # Create harmonic maps of the planet, excluding Y00
    # (lmax**2 maps, plus a negative version for all but Y00)
    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ind = 0
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):           
            planet.map[l, m] =  1.0
            sflux, lcs[ind]   = [a.eval() for a in system.flux(t, total=False)]
            planet.map[l, m] = -1.0
            sflux, lcs[ind+1] = [a.eval() for a in system.flux(t, total=False)]
            planet.map[l, m] = 0.0
            ind += 2
            
    # Run PCA to determine orthogonal light curves
    evalues, evectors, proj = pca.pca(lcs)

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)

    # Convert orthogonal light curves into maps
    eigeny = np.zeros((nharm, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps
    for j in range(nharm):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                eigeny[j,yi] = evectors.T[j,shi] - evectors.T[j,shi+1]
                yi  += 1
                shi += 2

    return eigeny, evalues, evectors, proj, lcs

def mkmaps(planet, eigeny, params, npar, ncurves, wl, rs, rp, ts, lat, lon):
    """
    Calculate flux maps and brightness temperature maps from
    2D map fits.

    Arguments
    ---------
    planet: starry Planet object
        Planet object. planet.map will be reset and modified within this
        function.

    eigeny: 2D array
        Eigenvalues for the eigenmaps that form the basis for the
        2D fit.

    params: 1D array
        Weights for each of the eigenmaps. Array length should be 
        (npar) * (number of wavelengths)

    npar: int
        Number of parameters in the fit per wavelength (e.g., a weight
        for each eigenmap, a base brightness, and a stellar correction term)

    ncurves: int
        Number of eigencurves (or eigenmaps) included in the total map.

    wl: 1D array
        The wavelengths for each 2D map, in microns.

    rs: float
        Radius of the star (same units as rp)

    rp: float
        radius of the planet (same units as rs)

    ts: float
        Temperature of the star in Kelvin

    lat: 2d array
        Latitudes of grid to calculate map

    lon: 2d array
        Longitudes of grid to calculate map

    Returns
    -------
    fmaps: 3D array
        Array with shape (nwl, nlat, nlon) of planetary emission at
        each wavelength and location

    tmaps: 3D array
        Same as fmaps but for brightness temperature.
    """
    nwl = len(wl)

    nlat, nlon = lat.shape
    
    fmaps = np.zeros((nwl, nlat, nlon)) # flux maps
    tmaps = np.zeros((nwl, nlat, nlon)) # temp maps

    # Convert wl to m
    wl_m = wl * 1e-6

    for j in range(nwl):
        planet.map[1:,:] = 0

        fmaps[j] = utils.mapintensity(planet.map, lat, lon,
                                      params[j*npar+ncurves])
        
        for i in range(ncurves):
            planet.map[1:,:] = eigeny[i,1:]
            fmaps[j] += utils.mapintensity(planet.map, lat, lon,
                                           params[j*npar+i])

        # Convert to brightness temperatures
        # see Rauscher et al., 2018, Eq. 8
        ptemp = (sc.h * sc.c) / (wl_m[j] * sc.k)
        sfact = 1 + params[j*npar+ncurves+1]
        tmaps[j] = ptemp / np.log(1 + (rp / rs)**2 *
                                  (np.exp(ptemp / ts) - 1) /
                                  (np.pi * fmaps[j] * sfact))

    return fmaps, tmaps

def emapminmax(planet, eigeny, ncurves):
    """
    Calculates the latitudes and longitudes of eigenmap minimum and maximum.
    Useful for checking for positivity in summed maps. Minimum is calculated
    with planet.map.minimize. Maximum is planet.map.minimize on a map
    with inverted sign eigenvalues.

    Arguments
    ---------
    planet: starry Planet object
        Planet object. planet.map will be modified in this function.

    eigeny: 2D array
        Array of eigenvalues for the eigenmaps. Same form as returned
        by mkcurves().

    ncurves: int
        Compute min and max for the first ncurves maps

    Returns
    -------
    lat: 1D array
        Array of latitudes, in degrees, of minimum and maximum of first
        ncurves maps. Length is 2 * ncurves

    lon: 1D array
        Array of longitudes, same format as lat.

    intens: 2D array
        Array of intensities at (lat, lon) for each eigenmap. Shape is
        (ncurves, nlocations).
    """
    lat    = np.zeros(2 * ncurves)
    lon    = np.zeros(2 * ncurves)
    intens = np.zeros((ncurves, len(lat)))

    nharm, ny = eigeny.shape
    
    lmax = np.int((nharm / 2 + 1)**0.5 - 1)

    # Find min/max locations of each eigenmap
    for j in range(ncurves):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j,yi]
                yi += 1

        lat[2*j], lon[2*j], _ = [a.eval() for a in planet.map.minimize()]

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = -1 * eigeny[j,yi]
                yi += 1        

        lat[2*j+1], lon[2*j+1], _ = [a.eval() for a in planet.map.minimize()]

    # Compute intensity of each eigenmap at EVERY position
    for j in range(ncurves):
        planet.map[1:,:] = 0

        yi = 1
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l, m] = eigeny[j,yi]
                yi += 1

        for i in range(len(lat)):
            intens[j,i] = planet.map.intensity(lat=lat[i], lon=lon[i]).eval()
            
    return lat, lon, intens

def intensities(planet, fit):
    wherevis = np.where((fit.lon + fit.dlon >= fit.minvislon) &
                        (fit.lon - fit.dlon <= fit.maxvislon))

    vislon = fit.lon[wherevis].flatten()
    vislat = fit.lat[wherevis].flatten()

    nloc = len(vislon)
    
    intens = np.zeros((fit.cfg.ncurves, nloc))

    for k in range(fit.cfg.ncurves):
        planet.map[1:,:] = 0
        yi = 1
        for l in range(1, fit.cfg.lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l,m] = fit.eigeny[k,yi]
                yi += 1

        intens[k] = planet.map.intensity(lat=vislat,
                                         lon=vislon).eval()

    return intens, vislat, vislon

            
            

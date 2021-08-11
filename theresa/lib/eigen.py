import numpy as np
import pca
import utils
import scipy.constants as sc

def mkcurves(system, t, lmax, y00, ncurves=None, method='pca'):
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

    y00: 1D array
        Light curve of a normalized, uniform map

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

    # Subtact uniform map contribution (starry includes this in all
    # light curves)
    lcs -= y00
            
    # Run PCA to determine orthogonal light curves
    if ncurves is None:
        ncurves = nharm
        
    evalues, evectors, proj = pca.pca(lcs, method=method, ncomp=ncurves)

    # Discard imaginary part of eigencurves to appease numpy
    proj = np.real(proj)

    # Convert orthogonal light curves into maps        
    eigeny = np.zeros((ncurves, (lmax + 1)**2))
    eigeny[:,0] = 1.0 # Y00 = 1 for all maps
    for j in range(ncurves):
        yi  = 1
        shi = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                # (ok because evectors has only been sorted along
                #  one dimension)
                eigeny[j,yi] = evectors.T[j,shi] - evectors.T[j,shi+1]
                yi  += 1
                shi += 2

    return eigeny, evalues, evectors, proj, lcs

def mkmaps(planet, eigeny, params, ncurves, wl, rs, rp, ts, lat, lon):
    """
    Calculate flux map and brightness temperature map from
    a single 2D map fit.

    Arguments
    ---------
    planet: starry Planet object
        Planet object. planet.map will be reset and modified within this
        function.

    eigeny: 2D array
        Eigenvalues for the eigenmaps that form the basis for the
        2D fit.

    params: 1D array
        Best-fitting parameters.

    ncurves: int
        Number of eigencurves (or eigenmaps) included in the total map.

    wl: 1D array
        The wavelength of the 2D map, in microns.

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
    fmap: 3D array
        Array with shape (nlat, nlon) of planetary emission at
        each wavelength and location

    tmap: 3D array
        Same as fmap but for brightness temperature.
    """
    nlat, nlon = lat.shape
    
    fmap = np.zeros((nlat, nlon)) # flux maps
    tmap = np.zeros((nlat, nlon)) # temp maps

    # Convert wl to m
    wl_m = wl * 1e-6

    planet.map[1:,:] = 0.0

    # Uniform map term
    fmap = utils.mapintensity(planet.map, lat, lon, params[ncurves])

    # Combine scaled eigenmap Ylm terms
    for i in range(ncurves):
        planet.map[1:,:] += eigeny[i,1:] * params[i]

    fmap += utils.mapintensity(planet.map, lat, lon, 1.0)

    # Subtract extra Y00 map that starry always includes
    planet.map[1:,:] = 0.0
    fmap -= utils.mapintensity(planet.map, lat, lon, 1.0)

    # Convert to brightness temperatures
    # see Rauscher et al., 2018, Eq. 8
    tmap = utils.fmap_to_tmap(fmap, wl_m, rp, rs, ts, params[ncurves+1])

    return fmap, tmap

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

def intensities(planet, fit, map):
    wherevis = np.where((fit.lon + fit.dlon >= fit.minvislon) &
                        (fit.lon - fit.dlon <= fit.maxvislon))

    vislon = fit.lon[wherevis].flatten()
    vislat = fit.lat[wherevis].flatten()

    nloc = len(vislon)
    
    intens = np.zeros((map.ncurves, nloc))

    for k in range(map.ncurves):
        planet.map[1:,:] = 0
        yi = 1
        for l in range(1, map.lmax + 1):
            for m in range(-l, l + 1):
                planet.map[l,m] = map.eigeny[k,yi]
                yi += 1

        intens[k] = planet.map.intensity(lat=vislat,
                                         lon=vislon).eval()
        planet.map[1:,:] = 0
        intens[k] -= planet.map.intensity(lat=vislat,
                                          lon=vislon).eval()

    return intens, vislat, vislon

            
            

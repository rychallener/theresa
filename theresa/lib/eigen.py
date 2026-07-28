import numpy as np
import jax
import pca
import time
import utils
import scipy.constants as sc
import jaxoplanet.starry.light_curves as light_curves

def mkcurves(fit, d, lmax, ncurves=None, method='pca',
             orbcheck=None, sigorb=None):
    """
    Generates light curves from a star+planet system at times t,
    for positive and negative spherical harmonics with l up to lmax.

    Arguments
    ---------
    system: object
        A jaxoplanet system object, initialized with a star and a planet

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
    t = d.t

    nt = len(t)

    def calcflux(y):
        star, planet, system = utils.initsystem(fit, lmax, y=y)
        lcfun = light_curves.light_curve(system, order=100)
        sflux, pflux = lcfun(t).T
        return sflux, pflux

    j_calcflux = jax.jit(calcflux)

    # Create harmonic maps of the planet, excluding Y00
    # (lmax**2 maps, plus a negative version for all but Y00)
    nharm = 2 * ((lmax + 1)**2 - 1)
    lcs = np.zeros((nharm, nt))
    ilc = 0
    for i, l in enumerate(range(1, lmax + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            # Create array of Ylm coefficients. The +1 includes Y00.
            y = np.zeros(nharm // 2 + 1)

            # initsystem does this for us, but just for clarity
            y[0] = 1.0
            # Set this specific harmonic to +1.0
            y[1 + ilc // 2] = 1.0
            
            sflux, lcs[ilc] = j_calcflux(y)
            lcs[ilc] -= d.pflux_y00

            # Insert negated version of the harmonic
            lcs[ilc+1] = -1 * np.copy(lcs[ilc])
            ilc += 2

    # If user wants to include additional eigencurves which explore
    # different orbital parameters
    if orbcheck is not None:
        # TODO: Implement orbcheck for jaxoplanet
        # For now, skip this feature
        print("Warning: orbcheck not yet implemented for jaxoplanet, skipping...")

    # Subtract uniform map contribution (jaxoplanet includes this in
    # all light curves)
    #lcs -= d.pflux_y00

    # Run PCA to determine orthogonal light curves
    if ncurves is None:
        ncurves = nharm
        if method == 'tsvd':
            ncurves -= 1

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

def mkmaps(fit, m, ln, params):
    """
    Calculate flux map and brightness temperature map from
    a single 2D map fit. Note that this function is simple and not
    optimized for speed. If you want to calculate a lot of maps,
    use utils.tmappost.

    Arguments
    ---------
    planet: jaxoplanet Surface object
        Planet object. planet.map will be reset and modified within this
        function.

    m: ThERESA Map object 
        (usually under fit.datasets[x].maps[x])

    ln: ThERESA LN object
        (usually under fit.datasets[x].maps[x].lxnx. For example, l1n2.)

    params: 1D Float array
        A set of parameters appropriate for the given LN object. For
        example, the parameters of the best-fitting model.

    Returns
    -------
    fmap: 1D/2D array
        Array with shape matching lat and lon of planetary emission at
        each wavelength and location

    tmap: 1D/2D array
        Same as fmap but for brightness temperature.
    """
    fmap = np.zeros(fit.lat.shape) # flux maps
    tmap = np.zeros(fit.lat.shape) # temp maps

    yval = np.zeros((ln.lmax+1)**2)
    yval[0] = 1.0

    for j in range(ln.ncurves):
        yval[1:] += params[j] * ln.eigeny[j,1:]

    star, planet, system = utils.initsystem(fit, ln.lmax, y=yval)

    # Non-uniform components
    fmap = planet.intensity(np.deg2rad(fit.lat),
                            np.deg2rad(fit.lon))

    # Fitted uniform component (-1 to remove default uniform
    # component). We could calculate this with a jaxoplanet object,
    # but it's faster to just use the knowledge that a uniform map has
    # Y00 / pi intensity everywhere.
    fmap += (params[ln.ncurves] - 1) / np.pi

    # Convert to brightness temperatures
    # see Rauscher et al., 2018, Eq. 8
    tmap = utils.fmap_to_tmap(fmap, m.wlmid, fit.cfg.planet.r,
                              fit.cfg.star.r, fit.cfg.star.t,
                              params[ln.ncurves+1],
                              starspec=fit.cfg.star.starspec,
                              fwl=m.filtwl, ftrans=m.filttrans,
                              swl=fit.starwl, sspec=fit.starflux)

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

def intensities(fit, data, ln):
    wherevis = np.where((fit.lon + fit.dlon >= data.minvislon) &
                        (fit.lon - fit.dlon <= data.maxvislon))

    vislon = fit.lon[wherevis].flatten()
    vislat = fit.lat[wherevis].flatten()

    nloc = len(vislon)
    
    intens = np.zeros((ln.ncurves, nloc))

    #Jit-ing this function could add speed, but this is pretty fast already.
    def evalintensity(yval):
        star, planet, system = utils.initsystem(fit, ln.lmax, y=yval)
        intensity = planet.intensity(np.deg2rad(vislat),
                                     np.deg2rad(vislon))

        # Couldn't we just subtract 1/pi here?
        star, planet, system = utils.initsystem(fit, ln.lmax)
        intensity -= planet.intensity(np.deg2rad(vislat),
                                      np.deg2rad(vislon))

        return intensity

    for k in range(ln.ncurves):
        intens[k] = evalintensity(ln.eigeny[k])
        
    return intens, vislat, vislon

            
            

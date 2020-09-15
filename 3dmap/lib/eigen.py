import numpy as np
import pca
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

def mkmaps(planet, eigeny, params, npar, wl, rs, rp, ts, proj='rect', res=300):
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
        Number of parameters in the fit per wavelenght (e.g., a weight
        for each eigenmap, a base brightness, and a stellar correction term)

    wl: 1D array
        The wavelengths for each 2D map, in microns.

    rs: float
        Radius of the star (same units as rp)

    rp: float
        radius of the planet (same units as rs)

    ts: float
        Temperature of the star in Kelvin

    res: int (optional)
        Resolution of the maps along each dimension. Default is 300.

    Returns
    -------
    fmaps: 3D array
        Array with shape (nwl, nres, nres) of planetary emission at
        each wavelength and location

    tmaps: 3D array
        Same as fmaps but for brightness temperature.
    """
    ncurves = int(len(params) / npar)

    fmaps = np.zeros((ncurves, res, res)) # flux maps
    tmaps = np.zeros((ncurves, res, res)) # temp maps

    # Convert wl to m
    wl_m = wl * 1e-6

    for j in range(len(wl)):
        planet.map[1:,:] = 0

        fmaps[j] = planet.map.render(theta=180, projection=proj,
                                    res=res).eval() * params[ncurves]

        for i in range(ncurves):
            planet.map[1:,:] = eigeny[i,1:]
            fmaps[j] += params[j*npar+i] * planet.map.render(theta=180,
                                                             projection=proj,
                                                             res=res).eval()
            
        # Convert to brightness temperatures
        # see Rauscher et al., 2018, Eq. 8
        ptemp = (sc.h * sc.c) / (wl_m[j] * sc.k)
        tmaps[j] = ptemp / np.log(1 + (rp / rs)**2 *
                                  (np.exp(ptemp / ts) - 1) /
                                  (np.pi * fmaps[j]))

    return fmaps, tmaps

        

import numpy as np
import pca

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
        "eigencurves"
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

    # Convert orthogonal light curves into a map
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
